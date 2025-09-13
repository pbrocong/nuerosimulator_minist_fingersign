import torch
import torch.optim as optim

class NeuroSimOptimizer(optim.Optimizer):
    """이상적 weight 업데이트를 '펄스 수 변화'로 바꾸어 적용"""
    def __init__(self, params, lr=1e-3, fitter=None, pulse_scaling_factor=1.0):
        if fitter is None: raise ValueError("fitter가 필요합니다.")
        self.fitter = fitter
        super().__init__(params, dict(lr=lr, pulse_scaling_factor=pulse_scaling_factor))

        self.pulse_states_ltp, self.pulse_states_ltd = {}, {}
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                g = p.data.clamp(fitter.target_min, fitter.target_max)
                self.pulse_states_ltp[pid] = fitter.p_of_g_ltp(g).nan_to_num(0.0)
                self.pulse_states_ltd[pid] = fitter.p_of_g_ltd(g).nan_to_num(0.0)

    @torch.no_grad()
    def step(self, closure=None):
        f = self.fitter
        for group in self.param_groups:
            lr, psf = group["lr"], group["pulse_scaling_factor"]
            for p in group["params"]:
                if p.grad is None: continue
                pid = id(p)
                grad, Gcur = p.grad.data, p.data
                Gtgt = (Gcur - lr * grad).clamp(f.target_min, f.target_max)
                ltp_mask, ltd_mask = grad < 0, grad > 0
                Pltp, Pltd = self.pulse_states_ltp[pid], self.pulse_states_ltd[pid]
                Gnew = Gcur.clone()

                if ltp_mask.any():
                    P_t = f.p_of_g_ltp(Gtgt[ltp_mask])
                    dP = (P_t - Pltp[ltp_mask]).nan_to_num(0.0)
                    dP_act = torch.floor(dP * psf + torch.rand_like(dP))
                    Pn = Pltp[ltp_mask] + dP_act
                    Gnew[ltp_mask] = f.g_of_p_ltp(Pn)

                if ltd_mask.any():
                    P_t = f.p_of_g_ltd(Gtgt[ltd_mask])
                    dP = (P_t - Pltd[ltd_mask]).nan_to_num(0.0)
                    dP_act = torch.floor(dP * psf + torch.rand_like(dP))
                    Pn = Pltd[ltd_mask] + dP_act
                    Gnew[ltd_mask] = f.g_of_p_ltd(Pn)

                p.data.copy_(Gnew.clamp(f.target_min, f.target_max))
                g = p.data
                self.pulse_states_ltp[pid] = f.p_of_g_ltp(g).nan_to_num(0.0)
                self.pulse_states_ltd[pid] = f.p_of_g_ltd(g).nan_to_num(0.0)