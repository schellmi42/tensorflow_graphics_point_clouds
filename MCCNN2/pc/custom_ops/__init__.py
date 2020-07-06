try:
  from .custom_ops_wrapper import basis_proj
  from .custom_ops_wrapper import build_grid_ds
  from .custom_ops_wrapper import compute_keys
  from .custom_ops_wrapper import compute_pdf
  from .custom_ops_wrapper import find_neighbors
  from .custom_ops_wrapper import sampling
except ImportError:
  # from .custom_ops_tf import basis_proj
  # from .custom_ops_tf import build_grid_ds
  from .custom_ops_tf import compute_keys_tf as compute_keys
  # from .custom_ops_tf import compute_pdf
  # from .custom_ops_tf import find_neighbors
  # from .custom_ops_tf import sampling
