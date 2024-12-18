"""solver.py"""

import os
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
import visdom
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataset import return_data
from model import BetaVAE_H, BetaVAE_B

warnings.filterwarnings("ignore")


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == "bernoulli":
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == "gaussian":
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(
            iter=[],
            recon_loss=[],
            total_kld=[],
            dim_wise_kld=[],
            mean_kld=[],
            mu=[],
            var=[],
            images=[],
        )

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        if args.dataset.lower() == "dsprites":
            self.nc = 1
            self.decoder_dist = "bernoulli"
        elif args.dataset.lower() == "3dchairs":
            self.nc = 3
            self.decoder_dist = "gaussian"
        elif args.dataset.lower() == "cars3d":
            self.nc = 3
            self.decoder_dist = "gaussian"
        elif args.dataset.lower() == "celeba":
            self.nc = 3
            self.decoder_dist = "gaussian"
        else:
            raise NotImplementedError

        if args.model == "H":
            net = BetaVAE_H
        elif args.model == "B":
            net = BetaVAE_B
        else:
            raise NotImplementedError("only support model H or B")

        self.net = net(self.z_dim, self.nc).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        self.scheduler = ReduceLROnPlateau(self.optim, mode="min", factor=0.5, patience=4, min_lr=1e-6)

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        self.win_lr = None
        self.win_beta = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather_by_ep = args.gather_by_ep
        if self.gather_by_ep is True:
            self.gather_step = 1
            self.display_step = len(self.data_loader)
            self.save_step = self.display_step
            self.max_iter = len(self.data_loader) * args.epoch

        if len(args.beta) > 1:
            self.beta_start = args.beta[0]
            self.beta_stop = args.beta[1]
            self.beta_step = self.max_iter / args.beta[2]
            self.beta = args.beta[0]
        else:
            self.beta = args.beta[0]

        self.gather = DataGather()

    def train(self):
        self.net_mode(train=True)
        self.C_max = torch.FloatTensor([self.C_max]).to(self.device)
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                x = x.to(self.device)
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist).to(self.device)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                self.beta = (
                    self.beta_start
                    - (self.beta_start - self.beta_stop)
                    * max(0, min(1, 2 * (self.global_iter % self.beta_step) / self.beta_step))
                    if hasattr(self, "beta_start")
                    else self.beta
                )

                if self.objective == "H":
                    beta_vae_loss = recon_loss + self.beta * total_kld
                elif self.objective == "B":
                    C = torch.clamp(self.C_max / self.C_stop_iter * self.global_iter, 0, self.C_max.item()).to(
                        self.device
                    )
                    beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(
                        iter=self.global_iter,
                        mu=mu.mean(0).detach(),
                        var=logvar.exp().mean(0).detach(),
                        recon_loss=recon_loss.detach(),
                        total_kld=total_kld.detach(),
                        dim_wise_kld=dim_wise_kld.detach(),
                        mean_kld=mean_kld.detach(),
                        lr=self.lr,
                        beta=self.beta,
                    )

                if self.global_iter % self.display_step == 0:
                    pbar.write(
                        "[{}] recon_loss: {:.3f} total_kld: {:.3f} mean_kld: {:.3f}".format(
                            self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()
                        )
                    )

                    var = logvar.exp().mean(0).detach()
                    var_str = ""
                    for j, var_j in enumerate(var):
                        var_str += "var_{}: {:.4f} ".format(j + 1, var_j)
                    pbar.write(var_str)

                    if self.objective == "B":
                        pbar.write("C: {:.3f}".format(C.item()))

                    if self.viz_on:
                        self.gather.insert(images=x.detach())
                        self.gather.insert(images=F.sigmoid(x_recon).detach())
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.gather.flush()

                    if self.viz_on or self.save_output:
                        self.viz_traverse()

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint("last")
                    pbar.write("Saved checkpoint (iter: {})".format(self.global_iter))

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break
            self.scheduler.step(beta_vae_loss)
        pbar.write("[Training Finished]")
        pbar.close()

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data["images"][0][:100].to(self.device)
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data["images"][1][:100].to(self.device)
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + "_reconstruction", opts=dict(title=str(self.global_iter)), nrow=10)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data["recon_loss"]).cpu()

        mus = torch.stack(self.gather.data["mu"]).cpu()
        vars = torch.stack(self.gather.data["var"]).cpu()

        dim_wise_klds = torch.stack(self.gather.data["dim_wise_kld"]).cpu()
        mean_klds = torch.stack(self.gather.data["mean_kld"]).cpu()
        total_klds = torch.stack(self.gather.data["total_kld"]).cpu()
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1)
        iters = torch.Tensor(self.gather.data["iter"])

        iters_lr = torch.Tensor(self.gather.data["iter"])
        lr = torch.Tensor(self.gather.data["lr"])
        beta = torch.Tensor(self.gather.data["beta"])

        x_label = "iteration"
        if self.gather_by_ep:
            recon_losses, mus, vars, dim_wise_klds, mean_klds, total_klds = map(
                lambda x: x.mean(0, keepdim=True), [recon_losses, mus, vars, dim_wise_klds, mean_klds, total_klds]
            )
            klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1)
            iters = torch.Tensor([iters[-1] / len(self.data_loader)])
            x_label = "epoch"

        legend = [f"z_{z_j}" for z_j in range(self.z_dim)] + ["mean", "total"]

        def plot_line(x, y, win_attr, title, legend_items=None):
            opts = dict(width=400, height=400, xlabel=x_label, title=title)
            if legend_items is not None:
                opts["legend"] = legend_items
            win = getattr(self, win_attr)
            update_mode = "append" if win else None
            setattr(
                self,
                win_attr,
                self.viz.line(X=x, Y=y, env=self.viz_name + "_lines", win=win, update=update_mode, opts=opts),
            )

        plot_line(iters, recon_losses, "win_recon", "reconstruction loss")
        plot_line(iters, klds, "win_kld", "kl divergence", legend)
        plot_line(iters, mus, "win_mu", "posterior mean", legend[: self.z_dim])
        plot_line(iters, vars, "win_var", "posterior variance", legend[: self.z_dim])
        plot_line(iters_lr, lr, "win_lr", "LR")
        plot_line(iters_lr, beta, "win_beta", "beta")

        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2 / 3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit + 0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets - 1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, : self.z_dim]

        random_z = torch.rand(1, self.z_dim).to(self.device)

        if self.dataset == "dsprites":
            fixed_idx1 = 87040  # square
            fixed_idx2 = 332800  # ellipse
            fixed_idx3 = 578560  # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1).to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, : self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2).to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, : self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3).to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, : self.z_dim]

            Z = {
                "fixed_square": fixed_img_z1,
                "fixed_ellipse": fixed_img_z2,
                "fixed_heart": fixed_img_z3,
                "random_img": random_img_z,
            }
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx).to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, : self.z_dim]

            Z = {"fixed_img": fixed_img_z, "random_img": random_img_z, "random_z": random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = "{}_latent_traversal(iter:{})".format(key, self.global_iter)

            if self.viz_on:
                self.viz.images(
                    samples, env=self.viz_name + "_traverse", opts=dict(title=title), nrow=len(interpolation)
                )

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 128, 128).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(
                        tensor=gifs[i][j].cpu(),
                        fp=os.path.join(output_dir, "{}_{}.jpg".format(key, j)),
                        nrow=self.z_dim,
                        pad_value=1,
                    )

                # grid2gif(os.path.join(output_dir, key + "*.jpg"), os.path.join(output_dir, key + ".gif"), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ("Only bool type is supported. True or False")

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {
            "net": self.net.state_dict(),
        }
        optim_states = {
            "optim": self.optim.state_dict(),
        }
        win_states = {
            "recon": self.win_recon,
            "kld": self.win_kld,
            "mu": self.win_mu,
            "var": self.win_var,
            "lr": self.win_lr,
            "beta": self.win_beta,
        }
        states = {
            "iter": self.global_iter,
            "win_states": win_states,
            "model_states": model_states,
            "optim_states": optim_states,
        }

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode="wb+") as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint["iter"]
            self.win_recon = checkpoint["win_states"]["recon"]
            self.win_kld = checkpoint["win_states"]["kld"]
            self.win_var = checkpoint["win_states"]["var"]
            self.win_mu = checkpoint["win_states"]["mu"]
            self.win_lr = checkpoint["win_states"]["lr"]
            self.win_beta = checkpoint["win_states"]["beta"]
            self.net.load_state_dict(checkpoint["model_states"]["net"])
            self.optim.load_state_dict(checkpoint["optim_states"]["optim"])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
