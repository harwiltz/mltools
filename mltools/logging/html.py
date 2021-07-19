import hashlib
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mltools.logging import ExperimentLogger

class HTMLExperimentLogger(ExperimentLogger):

    FILENAME = "index.html"

    def __init__(self,
                 exp_name,
                 root_path=".htmllogs",
                 template_path="./templates",
                 template_name="index.html",
                 **kwargs):
        super(HTMLExperimentLogger, self).__init__(exp_name, **kwargs)
        self.data = {
            "metrics": {},
            "metrics_rendered": {},
            "images": {},
            "plots": {},
            "text": {},
            "parameters": {},
        }

        self.rootdir = os.path.join(root_path, self.exp_name)
        os.makedirs(self.rootdir, exist_ok=True)
        self.filename = os.path.join(self.rootdir, HTMLExperimentLogger.FILENAME)

        self._env = Environment(
            loader=FileSystemLoader(template_path),
            autoescape=select_autoescape(['html'])
        )

        self._template = self._env.get_template(template_name)

    def build_page(self):
        with open(self.filename, 'w') as f:
            f.write(self._template.render(title=self.exp_name, epoch=self.epoch, data=self.data))

    def log_metric(self, tag, value, step, **kwargs):
        if tag not in self.data['metrics'].keys():
            self.data['metrics'][tag] = []
        self.data['metrics'][tag].append(value)
        metrics = self.data['metrics'][tag]
        plt.plot(metrics)
        fname = self._plot_fname(tag)
        plt.savefig(fname)
        plt.clf()
        self.data['metrics_rendered'][tag] = fname
        self.build_page()

    def log_image(self, tag: str, img: str, step: int, **kwargs):
        self.data['images'][tag] = (img, step)
        self.build_page()

    def log_plt(self, tag, fig, step, **kwargs):
        fname = self._plot_fname(tag)
        fig.savefig(fname)
        self.data['plots'][tag] = (fname, step)
        self.build_page()

    def log_text(self, tag, text, **kwargs):
        if tag not in self.data['text'].keys():
            self.data['text'][tag] = []
        self.data['text'][tag].append(text)
        self.build_page()

    def log_parameters(self, params, **kwargs):
        self.data['parameters'] = params

    def _plot_fname(self, tag):
        h = str(hash(tag)).encode()
        fname = hashlib.md5(h).hexdigest()
        return f"{os.path.join(self.rootdir, fname)}.png"
