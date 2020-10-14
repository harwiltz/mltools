from comet_ml import Experiment, OfflineExperiment

from mltools.logging import ExperimentLogger

class CometExperimentLogger(ExperimentLogger):
    def __init__(self, exp_name, online=True, **kwargs):
        super(CometExperimentLogger, self).__init__(exp_name, **kwargs)
        if online:
            self.comet = Experiment(**kwargs)
        else:
            self.comet = OfflineExperiment(**kwargs)

    def log_metric(self, tag, value, step, **kwargs):
        self.comet.log_metric(tag, value, **kwargs)

    def log_image(self, tag, img, step, **kwargs):
        self.comet.log_image(img, name=tag, step=step, **kwargs)

    def log_plt(self, tag, plt, step, **kwargs):
        self.comet.log_figure(figure=plt, figure_name=tag, step=step, **kwargs)

    def start_epoch(self, **kwargs):
        super(CometExperimentLogger, self).start_epoch()

    def end_epoch(self, **kwargs):
        super(CometExperimentLogger, self).end_epoch()
        self.comet.log_epoch_end(self.epoch, **kwargs)

    def end_experiment(self):
        self.comet.end()
