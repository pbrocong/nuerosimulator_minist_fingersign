import numpy as np
import torch
from scipy.optimize import curve_fit

class NeuroSimFitter:
    """LTP/LTD 곡선에 NeuroSim 형태로 피팅하고
    P<->G 변환(g_of_p, p_of_g) 함수를 제공"""
    def __init__(self, ltp_df, ltd_df, target_range=(-1.0, 1.0),
                 ltp_fit_ratio=1.0, ltd_fit_ratio=1.0):
        # 전체(real) 값
        self.ltp_p_full = ltp_df["PulseNum"].values
        self.ltd_p_full = ltd_df["PulseNum"].values
        ltp_g_real_all = ltp_df["Conductance"].values
        ltd_g_real_all = ltd_df["Conductance"].values

        self.g_min_real = min(ltp_g_real_all.min(), ltd_g_real_all.min())
        self.g_max_real = max(ltp_g_real_all.max(), ltd_g_real_all.max())
        self.target_min, self.target_max = target_range

        # 안정 구간 비율 사용
        li = int(len(ltp_df) * (1 - ltp_fit_ratio))
        di = int(len(ltd_df) * (1 - ltd_fit_ratio))
        self.ltp_p_fit = ltp_df["PulseNum"].values[li:]
        ltp_g_fit_real = ltp_df["Conductance"].values[li:]
        self.ltd_p_fit = ltd_df["PulseNum"].values[di:]
        ltd_g_fit_real = ltd_df["Conductance"].values[di:]

        # 스케일링
        self.ltp_g_scaled = self.scale(ltp_g_fit_real)
        self.ltd_g_scaled = self.scale(ltd_g_fit_real)
        self.g_min_fit_scaled = self.scale(ltp_g_fit_real[0]) if len(ltp_g_fit_real) else self.target_min
        self.g_max_fit_scaled = self.scale(ltd_g_fit_real[0]) if len(ltd_g_fit_real) else self.target_max

        # 피팅
        self.A_LTP, self.B_LTP = self._fit_ltp()
        self.A_LTD, self.B_LTD = self._fit_ltd()
        if None in (self.A_LTP, self.A_LTD):
            raise RuntimeError("NeuroSim 모델 피팅 실패")

    # ----- 스케일/역스케일 -----
    def scale(self, g_real):
        denom = (self.g_max_real - self.g_min_real)
        if denom == 0: return np.zeros_like(g_real) + self.target_min
        return ((g_real - self.g_min_real) / denom) * (self.target_max - self.target_min) + self.target_min

    def unscale(self, g_scaled):
        denom = (self.target_max - self.target_min)
        if denom == 0: return np.zeros_like(g_scaled) + self.g_min_real
        return ((g_scaled - self.target_min) / denom) * (self.g_max_real - self.g_min_real) + self.g_min_real

    # ----- 곡선 피팅 -----
    def _fit_ltp(self):
        if len(self.ltp_p_fit) < 2: return 1.0, 1.0
        def model(P, A, B):  # (P-start) 모델
            return B * (1 - np.exp(-(P - self.ltp_p_fit[0]) / A)) + self.g_min_fit_scaled
        p0 = [self.ltp_p_fit.mean(), self.target_max - self.g_min_fit_scaled]
        try:
            (A, B), _ = curve_fit(model, self.ltp_p_fit, self.ltp_g_scaled, p0=p0, maxfev=10000)
            return float(A), float(B)
        except Exception:
            return None, None

    def _fit_ltd(self):
        if len(self.ltd_p_fit) < 2: return 1.0, 1.0
        def model(P, A, B):
            return -B * (1 - np.exp(-(P - self.ltd_p_fit[0]) / A)) + self.g_max_fit_scaled
        p0 = [self.ltd_p_fit.mean(), self.g_max_fit_scaled - self.target_min]
        try:
            (A, B), _ = curve_fit(model, self.ltd_p_fit, self.ltd_g_scaled, p0=p0, maxfev=10000)
            return float(A), float(B)
        except Exception:
            return None, None

    # ----- P<->G 변환 (torch tensor 버전) -----
    def g_of_p_ltp(self, P):
        p0 = self.ltp_p_fit[0] if len(self.ltp_p_fit) else 0
        effP = (P - p0).clamp(min=0)
        return self.B_LTP * (1 - torch.exp(-effP / (self.A_LTP + 1e-9))) + self.g_min_fit_scaled

    def p_of_g_ltp(self, G):
        eps = 1e-9
        arg = 1 - (G - self.g_min_fit_scaled) / (self.B_LTP + eps)
        dP = -(self.A_LTP) * torch.log(arg.clamp(min=eps))
        return dP + (self.ltp_p_fit[0] if len(self.ltp_p_fit) else 0)

    def g_of_p_ltd(self, P):
        p0 = self.ltd_p_fit[0] if len(self.ltd_p_fit) else 0
        effP = (P - p0).clamp(min=0)
        return -self.B_LTD * (1 - torch.exp(-effP / (self.A_LTD + 1e-9))) + self.g_max_fit_scaled

    def p_of_g_ltd(self, G):
        eps = 1e-9
        arg = 1 - (self.g_max_fit_scaled - G) / (self.B_LTD + eps)
        dP = -(self.A_LTD) * torch.log(arg.clamp(min=eps))
        return dP + (self.ltd_p_fit[0] if len(self.ltd_p_fit) else 0)