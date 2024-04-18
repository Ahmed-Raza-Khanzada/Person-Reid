def ssim_score(matrix1, matrix2):
  if matrix1.shape != matrix2.shape:
    raise ValueError("Matrices must have the same shape")
  matrix1 = matrix1.astype(np.float64)
  matrix2 = matrix2.astype(np.float64)
  score = ssim(matrix1, matrix2)

  return score


def feature_score(matrix1, matrix2):
  if matrix1.shape != matrix2.shape:
    raise ValueError("Matrices must have the same shape")
  diff = np.abs(matrix1 - matrix2)
  mean_diff = np.mean(diff)
  std_dev = np.std(diff)
  score = 1 - (mean_diff / std_dev)
  return score