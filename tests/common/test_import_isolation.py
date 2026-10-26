"""The pure distribution can be imported without Meep installed."""

import subprocess
import sys
import textwrap


def run(code):
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_import_without_meep_and_explicit_selection():
    run("""
        import importlib.abc
        import sys

        class BlockMeep(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'meep':
                    raise ModuleNotFoundError('unavailable engine', name=fullname)

        sys.meta_path.insert(0, BlockMeep())
        import tama
        assert not {'meep', 'numpy'} & sys.modules.keys()
        assert 'TDAObjective' in dir(tama)
        for invalid in (None, [], 1, 'Meep', 'auto', 'unknown_backend', ''):
            try:
                tama.get_backend(invalid)
            except ValueError as exc:
                assert "'meep'" in str(exc)
            else:
                raise AssertionError(invalid)
        for access, missing in (
            (lambda: tama.get_backend('meep'), {'meep'}),
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


def test_native_sampler_import_guard_raises_clear_missing_extension_error():
    run("""
        import importlib
        import importlib.abc
        import sys

        class BlockNative(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'tama_meep_native':
                    raise ModuleNotFoundError('unavailable extension', name=fullname)

        sys.meta_path.insert(0, BlockNative())
        try:
            importlib.import_module('tama.native_sampler')
        except ImportError as exc:
            assert 'native sampler extension is not built' in str(exc)
            assert 'install tama-meep-native' in str(exc)
        else:
            raise AssertionError('missing native extension must fail')
    """)
