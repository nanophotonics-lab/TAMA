"""The pure distribution must be usable before either engine is installed."""

import subprocess
import sys
import textwrap


def run(code):
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_import_without_engines_and_explicit_selection():
    run("""
        import importlib.abc
        import sys

        class BlockEngines(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'meep', 'fdtdx', 'jax', 'equinox', 'tama_fdtdx'}:
                    raise ModuleNotFoundError('unavailable engine', name=fullname)

        sys.meta_path.insert(0, BlockEngines())
        import tama
        assert not {'meep', 'fdtdx', 'jax', 'equinox', 'numpy', 'tama_fdtdx'} & sys.modules.keys()
        assert 'TDAObjective' in dir(tama)
        for invalid in (None, [], 1, 'Meep', 'auto', ''):
            try:
                tama.get_backend(invalid)
            except ValueError as exc:
                assert "'meep' or 'fdtdx'" in str(exc)
            else:
                raise AssertionError(invalid)
        for access, missing in (
            (lambda: tama.get_backend('meep'), {'meep'}),
            (lambda: tama.get_backend('fdtdx'), {'fdtdx', 'jax', 'equinox'}),
            (lambda: tama.PointTarget, {'meep'}),
        ):
            try:
                access()
            except ModuleNotFoundError as exc:
                assert exc.name in missing, exc
                assert 'backend' in str(exc), exc
            else:
                raise AssertionError(missing)
        try:
            tama.nonexistent_attribute
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown attribute must fail')
    """)


def test_unrelated_dependency_error_is_not_hidden():
    run("""
        import importlib.abc
        import sys

        class BlockNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'numpy':
                    raise ModuleNotFoundError('original dependency failure', name=fullname)

        sys.meta_path.insert(0, BlockNumpy())
        import tama
        try:
            tama.tanh_projection
        except ModuleNotFoundError as exc:
            assert exc.name == 'numpy'
            assert str(exc) == 'original dependency failure'
        else:
            raise AssertionError('unrelated dependency error was hidden')
    """)


def test_fdtdx_engine_itself_is_required_at_selection():
    run("""
        import importlib.abc
        import sys

        class BlockFdtdx(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'fdtdx':
                    raise ModuleNotFoundError('missing FDTDX engine', name=fullname)

        sys.meta_path.insert(0, BlockFdtdx())
        import tama
        try:
            tama.get_backend('fdtdx')
        except ModuleNotFoundError as exc:
            assert exc.name == 'fdtdx', exc
            assert 'tama[fdtdx]' in str(exc), exc
        else:
            raise AssertionError('backend selection accepted a missing FDTDX engine')
    """)
