"""Regression: a notebook that ran but displayed zero figures must fail."""
import base64
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from lecture_notebook_validation import output_issues

PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY1sAAAAASUVORK5CYII='


def notebook(outputs):
    return {'cells':[{'id':'dataset','cell_type':'code','execution_count':1,'outputs':outputs}]}


class NotebookOutputTests(unittest.TestCase):
    def test_no_images_fails_even_if_executed(self):
        issues=output_issues(notebook([{'output_type':'stream','text':'completed'}]),'00_environment_check.ipynb')
        self.assertTrue(any('embedded PNG' in s for s in issues))

    def test_inline_png_passes(self):
        self.assertEqual(output_issues(notebook([{'output_type':'display_data',
            'data':{'image/png':PNG}}]),'00_environment_check.ipynb'),[])

    def test_agg_warning_fails(self):
        outputs=[{'output_type':'display_data','data':{'image/png':PNG}},
                 {'output_type':'stream','text':['FigureCanvasAgg is non-interactive, and thus cannot be shown']}]
        self.assertTrue(any('warning' in s for s in output_issues(notebook(outputs),'00_environment_check.ipynb')))

    def test_invalid_png_fails(self):
        outputs=[{'output_type':'display_data','data':{'image/png':base64.b64encode(b'not png').decode()}}]
        self.assertTrue(any('malformed' in s for s in output_issues(notebook(outputs),'00_environment_check.ipynb')))


if __name__=='__main__': unittest.main()
