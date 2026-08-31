import torch
import torch.nn as nn
import torch.nn.functional as F

class OtLoss(nn.Module):
    def __init__(self, dist_metric='cosine', reg=0.01, max_iter=50, eps=1e-8, log_domain=False):
        super(OtLoss, self).__init__()
        self.dist_metric = dist_metric
        self.reg = reg
        self.max_iter = max_iter
        self.eps = eps
        self.log_domain = log_domain   # If True, use the log-sum-exp stabilized Sinkhorn (avoids underflow)

    def _get_cost_matrix(self, x, y):
        if self.dist_metric == 'cosine':
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            cosine_sim = torch.matmul(x_norm, y_norm.T)
            cosine_dist = 1.0 - cosine_sim
            return cosine_dist
        
        elif self.dist_metric == 'hamming':
            x_bin = ((x + 1) // 2).to(torch.uint8)
            y_bin = ((y + 1) // 2).to(torch.uint8)
            x_bin = x_bin.unsqueeze(1)
            y_bin = y_bin.unsqueeze(0)
            xor = (x_bin != y_bin).float()
            return xor.sum(dim=2)

    def _sinkhorn(self, C, mu, nu):

        # Initialize Sinkhorn variables
        K = torch.exp(-C / self.reg)  # Kernel matrix, Shape: (N, M)
        u = torch.ones_like(mu)      # Shape: (N,)
        v = torch.ones_like(nu)      # Shape: (M,)

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            v = nu / (torch.matmul(K.T, u) + self.eps)  # v <- nu / (K^T u)
            u = mu / (torch.matmul(K, v) + self.eps)   # u <- mu / (K v)

        # Calculate the approximate optimal transport plan P
        # P = diag(u) @ K @ diag(v)
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        return P

    def _sinkhorn_log(self, C, mu, nu):
        """log-domain (log-sum-exp) Sinkhorn. Avoids exp(-C/reg) underflow so that
        K'>0 effectively holds even for a small reg / large mismatch penalty.
        Mathematically the same iteration as naive _sinkhorn, only numerically stable."""
        logmu = torch.log(mu)
        lognu = torch.log(nu)
        a = torch.zeros_like(mu)   # a = log u
        b = torch.zeros_like(nu)   # b = log v
        Cl = -C / self.reg
        for _ in range(self.max_iter):
            b = lognu - torch.logsumexp(Cl + a.unsqueeze(1), dim=0)
            a = logmu - torch.logsumexp(Cl + b.unsqueeze(0), dim=1)
        logP = a.unsqueeze(1) + Cl + b.unsqueeze(0)
        return torch.exp(logP)

    def _solve(self, C, mu, nu):
        """Select the stabilized or the basic Sinkhorn according to the log_domain flag."""
        return self._sinkhorn_log(C, mu, nu) if self.log_domain else self._sinkhorn(C, mu, nu)

    def forward(self, source, target):
        """Computes Sinkhorn divergence (entropic regularized OT cost)."""
        N, D_s = source.shape
        M, D_t = target.shape
        if D_s != D_t:
            raise ValueError(f"Source and Target dimensions must match: {D_s} != {D_t}")

        # Calculate the base cost matrix
        C = self._get_cost_matrix(source, target)  # Shape: (N, M)

        # Create marginal distributions (uniform)
        mu = torch.full((N,), 1.0 / N, dtype=source.dtype, device=source.device)
        nu = torch.full((M,), 1.0 / M, dtype=target.dtype, device=target.device)

        # Get transport plan from the core solver
        P = self._solve(C, mu, nu)

        # Compute the Sinkhorn loss (entropic regularized transport cost)
        loss = torch.sum(P * C)
        return loss, P

class NormalOtLoss(OtLoss):
    """
    Inherits from OtLoss but keeps the same forward signature as MaskedOtLoss.
    It takes (source, target, source_labels, target_labels) as arguments, but
    ignores the label arguments and computes the parent class's (non-masked) OT loss.

    Convenient for use in ablation studies.
    """
    def __init__(self, dist_metric='cosine', reg=0.01, max_iter=50, eps=1e-8):
        # Call the parent class (OtLoss) init as-is.
        super().__init__(dist_metric=dist_metric, reg=reg, max_iter=max_iter, eps=eps)
        # mismatch_penalty is not needed in this class.

    def forward(self, source, target, source_labels, target_labels):
        """
        Takes the same 4 arguments as MaskedOtLoss, but
        deliberately ignores source_labels and target_labels and
        calls the parent class (OtLoss) forward(source, target).
        """
        # The parent class (OtLoss) forward takes only 2 arguments.
        return super().forward(source, target)


class MaskedOtLoss(OtLoss):
    def __init__(self, dist_metric='cosine', reg=0.01, max_iter=50, eps=1e-8,
                 mismatch_penalty=1e9, log_domain=False):
        super().__init__(dist_metric=dist_metric, reg=reg, max_iter=max_iter, eps=eps,
                         log_domain=log_domain)
        self.mismatch_penalty = mismatch_penalty

    def _create_label_mask(self, source_labels, target_labels):
        if source_labels.ndim == 1 and target_labels.ndim == 1:
            labels_match = source_labels.unsqueeze(1) == target_labels.unsqueeze(0)
            mask = (~labels_match).float()
        else:
            raise ValueError("Unsupported label dimensions.")
        return mask

    def forward(self, source, target, source_labels, target_labels):
        N, D_s = source.shape
        M, D_t = target.shape

        if N != source_labels.shape[0]:
            raise ValueError(f"Number of source samples ({N}) must match number of source labels ({source_labels.shape[0]})")
        if M != target_labels.shape[0]:
            raise ValueError(f"Number of target samples ({M}) must match number of target labels ({target_labels.shape[0]})")
        if D_s != D_t:
            raise ValueError(f"Source and Target dimensions must match for Wasserstein: {D_s} != {D_t}")
        
        # --- Get Marginal Distributions ---
        mu = torch.full((N,), 1.0 / N, dtype=source.dtype, device=source.device)
        nu = torch.full((M,), 1.0 / M, dtype=target.dtype, device=target.device)

        # 1. Calculate the original cost matrix (reusing parent's method)
        C_original = self._get_cost_matrix(source, target)  # Shape: (N, M)

        # 2. Create the masked cost matrix for Sinkhorn calculation
        label_mask = self._create_label_mask(source_labels, target_labels)  # Shape (N, M)
        C_masked = C_original + self.mismatch_penalty * label_mask

        # 3. Run Sinkhorn algorithm using the masked cost matrix (reusing parent's core solver)
        #    If log_domain=True, use the log-sum-exp stabilized version (avoids underflow)
        P = self._solve(C_masked, mu, nu)

        # 4. Compute the final loss using the *original* cost matrix C_original
        #    This measures the actual cost of the transport plan found.
        loss = torch.sum(P * C_original)
        transport_plan = P

        return loss, transport_plan
    
class MaskedOtLoss_jc(OtLoss):
    def __init__(self, dist_metric='cosine', reg=0.01, max_iter=50, eps=1e-8, mismatch_penalty=1e9):
        super().__init__(dist_metric=dist_metric, reg=reg, max_iter=max_iter, eps=eps)
        self.mismatch_penalty = mismatch_penalty

    def _create_label_penalty_coeffs(self, source_labels, target_labels):
        N = source_labels.shape[0]
        M = target_labels.shape[0]
        
        penalty_coeffs = torch.zeros(N, M, device=source_labels.device, dtype=torch.float32)

        # Case 1: both are 1D (single-label)
        if source_labels.ndim == 1 and target_labels.ndim == 1:
            labels_match = source_labels.unsqueeze(1) == target_labels.unsqueeze(0)
            penalty_coeffs = (~labels_match).float()

        # Case 2: both are 2D (multi-label, multi-hot)
        elif source_labels.ndim == 2 and target_labels.ndim == 2:
            s_labels_float = source_labels.float()
            t_labels_float = target_labels.float()
            intersect = torch.matmul(s_labels_float, t_labels_float.T)  # (N, M)
            sum_s = s_labels_float.sum(dim=1, keepdim=True)  # (N, 1)
            sum_t = t_labels_float.sum(dim=1, keepdim=True).T  # (1, M)
            union = sum_s + sum_t - intersect
            jaccard_similarity = intersect / (union + self.eps) # Add eps for stability
            penalty_coeffs = 1.0 - jaccard_similarity

        # Case 3: source is multi-label, target is single-label
        elif source_labels.ndim == 2 and target_labels.ndim == 1:
            num_classes = source_labels.shape[1]
            s_labels_float = source_labels.float()
            t_onehot = F.one_hot(target_labels, num_classes=num_classes).float()
            
            intersect = torch.matmul(s_labels_float, t_onehot.T) # (N, M)
            sum_s = s_labels_float.sum(dim=1, keepdim=True)
            sum_t = t_onehot.sum(dim=1, keepdim=True).T
            union = sum_s + sum_t - intersect
            jaccard_similarity = intersect / (union + self.eps)
            penalty_coeffs = 1.0 - jaccard_similarity
            
        # Case 4: source is single-label, target is multi-label
        elif source_labels.ndim == 1 and target_labels.ndim == 2:
            num_classes = target_labels.shape[1]
            s_onehot = F.one_hot(source_labels, num_classes=num_classes).float()
            t_labels_float = target_labels.float()

            intersect = torch.matmul(s_onehot, t_labels_float.T) # (N, M)
            sum_s = s_onehot.sum(dim=1, keepdim=True)
            sum_t = t_labels_float.sum(dim=1, keepdim=True).T
            union = sum_s + sum_t - intersect
            jaccard_similarity = intersect / (union + self.eps)
            penalty_coeffs = 1.0 - jaccard_similarity
            
        else:
            raise ValueError(f"Unsupported label dimensions. Source: {source_labels.ndim}D, Target: {target_labels.ndim}D")

        # Ensure penalty_coeffs are between 0 and 1, and handle potential NaNs from division by zero if eps is not enough
        penalty_coeffs = torch.clamp(penalty_coeffs, 0.0, 1.0)
        penalty_coeffs[torch.isnan(penalty_coeffs)] = 1.0 # Default to max penalty if NaN (e.g. union is 0)
        return penalty_coeffs

    def forward(self, source, target, source_labels, target_labels):
        N, D_s = source.shape
        M, D_t = target.shape

        if N == 0 or M == 0: # Handle empty inputs
            return torch.tensor(0.0, device=source.device, dtype=source.dtype), torch.zeros(N, M, device=source.device, dtype=source.dtype)


        if N != source_labels.shape[0]:
            raise ValueError(f"Number of source samples ({N}) must match number of source labels ({source_labels.shape[0]})")
        if M != target_labels.shape[0]:
            raise ValueError(f"Number of target samples ({M}) must match number of target labels ({target_labels.shape[0]})")

        # Create the label penalty coefficients (0 to 1 scale)
        label_penalty_coeffs = self._create_label_penalty_coeffs(source_labels, target_labels) # Shape (N, M)

        mu = torch.full((N,), 1.0 / N, dtype=source.dtype, device=source.device)
        nu = torch.full((M,), 1.0 / M, dtype=target.dtype, device=target.device)

        if D_s != D_t:
            raise ValueError(f"Source and Target dimensions must match for Wasserstein: {D_s} != {D_t}")
        C_original = self._get_cost_matrix(source, target)
        
        # Apply scaled penalty
        C_masked = C_original + self.mismatch_penalty * label_penalty_coeffs
        K = torch.exp(-C_masked / self.reg)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for _ in range(self.max_iter):
            v_prev = v
            u_prev = u
            # Add self.eps in denominator to prevent division by zero if K.T @ u or K @ v is zero.
            v = nu / (torch.matmul(K.T, u) + self.eps) 
            u = mu / (torch.matmul(K, v) + self.eps)
            if torch.isnan(u).any() or torch.isnan(v).any():
                # print("Warning: NaN detected in Sinkhorn u/v. Reverting to previous iterate.")
                u = u_prev
                v = v_prev
                break 
            # Optional: Convergence check (e.g., torch.sum(torch.abs(v - v_prev)) < some_tol)
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        loss = torch.sum(P * C_original)
        transport_plan = P

        return loss, transport_plan