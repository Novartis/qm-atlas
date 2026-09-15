import logging
import multiprocessing
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

from rdkit import Chem
from tqdm import tqdm

from qm_atlas.utils import meta_func

_logger = logging.getLogger(__name__)


TQDM_OPTIONS: dict[str, Any] = {
    "ncols": 80,
}


@contextmanager
def pickle_mol_properties(properties: Chem.PropertyPickleOptions | None):
    """Context manager to temporarily change RDKit pickle properties. Restores original settings after the block.
    Changing rdkit's pickle properties can be useful when handling molecules in multiprocessing, where they are pickled.
    With the default rdkit settings, the properties are lost in this process, which may be undesirable.

    Args:
        properties (PropertyPickleOptions | None):
            PropertyPickleOptions value (e.g., Chem.PropertyPickleOptions.AllProps),
            or None. If None is passed, the context manager does nothing.
    """
    if properties is None:
        yield
        return

    # Save current settings
    original_props = Chem.GetDefaultPickleProperties()
    try:
        # Set new properties
        Chem.SetDefaultPickleProperties(properties)
        yield
    finally:
        # Always restore original settings
        Chem.SetDefaultPickleProperties(original_props)


# TODO: move this back to ppqm
def run_parallel(
    func: Callable,
    arg_list: Sequence[tuple],
    show_progress: bool = True,
    n_cores: int = 1,
    n_jobs: int | None = None,
    title: str = "Parallel",
    func_has_ncores_arg: bool = False,
    rdkit_pickle_properties: Chem.PropertyPickleOptions | None = None,
    **kwargs,
) -> list:
    """Run *func* over *arg_list* in parallel (or serially for a single core/item).

    Use functools.partial to bind additional constant arguments before calling.

    Example:
        >>> def most_important(x, y, z, add=5):
        ...     return x + y + z + add
        >>> results = run_parallel(most_important, [(1,), (2,)], n_cores=2)

    Args:
        func (Callable): The function to apply to each entry of *arg_list*.
        arg_list (Sequence[tuple]): The per-call positional arguments.
        show_progress (bool): If True, display a tqdm progress bar. Defaults to True.
        n_cores (int): Number of worker processes; <=1 runs serially. Defaults to 1.
        n_jobs (int | None): Total number of jobs for the progress bar. Defaults to
            len(arg_list) when None.
        title (str): Label shown on the progress bar. Defaults to "Parallel".
        func_has_ncores_arg (bool): If True, an ``n_cores`` kwarg is passed to *func*,
            split across the workers. Defaults to False.
        rdkit_pickle_properties (Chem.PropertyPickleOptions | None): RDKit pickle
            properties to preserve when sending molecules to workers. Defaults to None.
        **kwargs: Additional keyword arguments bound to *func* for every call.

    Returns:
        list: The results, in the same order as *arg_list*.
    """
    if func_has_ncores_arg:
        kwargs["n_cores"] = n_cores // len(arg_list) or 1

    _func = partial(meta_func, func, **kwargs)

    pbar = None

    if show_progress:

        if n_jobs is None:
            n_jobs = len(arg_list)

        pbar = tqdm(
            total=n_jobs,
            desc=f"{title}({n_cores})",
            **TQDM_OPTIONS,
        )

    results = []

    if n_cores <= 1 or len(arg_list) <= 1:
        for arg in arg_list:
            result = _func(arg)

            if pbar:
                pbar.update(1)

            results.append(result)
        return results

    with pickle_mol_properties(rdkit_pickle_properties):
        p = multiprocessing.Pool(processes=n_cores)
        try:
            results_iter = p.imap(_func, arg_list, chunksize=1)

            for result in results_iter:

                if pbar:
                    pbar.update(1)

                results.append(result)

        except KeyboardInterrupt as exc:
            _logger.error("got ^C while running pool of workers...")
            p.terminate()
            raise exc

        except Exception as e:
            _logger.error(f"got exception: {e}, terminating the pool")
            p.terminate()
            raise e

        finally:
            p.terminate()

        if pbar:
            pbar.close()

    return results
