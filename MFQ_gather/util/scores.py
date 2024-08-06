import torch
import torch.jit

@torch.jit.script
def compute_silhouette_score(points: torch.Tensor, labels: torch.Tensor)-> torch.Tensor:
    """计算轮廓系数 tensor

    Args:
        points: batch_size x n x d, n是样本数, d是特征数
        labels: batch_size x n, 每个样本的类别

    Returns:
        scores: batch_size, 轮廓系数
    """
    batch_size, n, _ = points.size()
    r_silhouette = torch.zeros(batch_size, dtype=torch.float32, device=points.device)
    for i in range(batch_size):
        pts = points[i]
        lbs = labels[i]
        
        # t0 = time.time()
        unique_labels = torch.unique(lbs)

        # 计算所有点之间的距离
        distances = torch.cdist(pts, pts)

        # 计算 a(i)
        a = torch.zeros(n, dtype=torch.float32).to(pts.device)
        for label in unique_labels:
            mask = (lbs == label)
            if mask.sum() > 1:
                a[mask] = torch.mean(distances[mask][:, mask], dim=1)
        
        # 计算 b(i)
        b = torch.full((n,), float('inf'), dtype=torch.float32).to(pts.device)
        for label in unique_labels:
            mask = (lbs == label)
            if mask.sum() > 0:
                other_labels = unique_labels[unique_labels != label]
                if len(other_labels) > 0:
                    other_cluster_dists = torch.stack([torch.mean(distances[mask][:, lbs == other_label], dim=1) for other_label in other_labels])
                    b[mask] = torch.min(other_cluster_dists, dim=0)[0]

        
        s = (b - a) / torch.maximum(a, b)
        s[torch.maximum(a, b) == 0] = 0  # 避免分母为0的情况
        r_silhouette[i] = torch.mean(s)
        # t1 = time.time()
        # print(f"Time: {t1 - t0}")        

    return r_silhouette
@torch.jit.script
def max_silhouette_score(points, labels):
    """计算最大的轮廓系数对应的k   tensor

    Args:
        points: batch_size x num_k x n x d, n是样本数, d是特征数, num_k是k的个数
        labels: batch_size x num_k x n, 每个样本的类别

    Returns:
        scores: batch_size, 轮廓系数
    """

    batch_size = points.size(0)
    max_k = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(batch_size):
        r_silhouette = compute_silhouette_score(points[i], labels[i])
        max_k[i] = torch.argmax(r_silhouette)

    return max_k
