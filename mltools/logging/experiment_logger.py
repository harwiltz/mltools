class ExperimentLogger(object):
    def __init__(self, exp_name, tags=[], **kwargs):
        self.exp_name = exp_name
        self.epoch = 0
        self.tags = tags

    def instantiate(logger_type, exp_name, **kwargs):
        logger_type = logger_type.lower()
        if (logger_type == 'default') or (logger_type == 'cli'):
            return CLIExperimentLogger(exp_name, **kwargs)
        if logger_type == 'comet':
            return CometExperimentLogger(exp_name, **kwargs)

    def log_metric(self, tag, value, step, **kwargs):
        raise NotImplementedError

    def log_image(self, tag, img, step, **kwargs):
        raise NotImplementedError

    def log_plt(self, tag, plt, step, **kwargs):
        raise NotImplementedError

    def log_text(self, tag, text, **kwargs):
        raise NotImplementedError

    def log_parameters(self, params, **kwargs):
        raise NotImplementedError

    def start_epoch(self, **kwargs):
        pass

    def end_epoch(self, **kwargs):
        self.epoch += 1

    def end_experiment(self):
        raise NotImplementedError

class CLIExperimentLogger(ExperimentLogger):
    def __init__(self, exp_name, **kwargs):
        super(CLIExperimentLogger, self).__init__(exp_name, **kwargs)
        self._has_warned_img = False
        self._has_warned_plt = False
        print(f"Instantiated logger for experiment \"{exp_name}\"")

    def log_metric(self, tag, value, step, **kwargs):
        print("[METRIC] {:>32} ({:>5}): {:>16.11f}".format(tag, step, value))

    def log_image(self, tag, img, step, **kwargs):
        if self._has_warned_img:
            return
        else:
            warn_message_pre = f"[  WARN] {self.__class__} does not support image logging"
            warn_message_suf = f"skipping log of {tag} ({step})"
            print(f"{warn_message_pre} -- {warn_message_suf}")
            self._has_warned_img = True

    def log_plt(self, tag, plt, step, **kwargs):
        if self._has_warned_plt:
            return
        else:
            warn_message_pre = f"[  WARN] {self.__class__} does not support plt logging"
            warn_message_suf = f"skipping log of {tag} ({step})"
            print(f"{warn_message_pre} -- {warn_message_suf}")
            self._has_warned_plt = True

    def log_text(self, tag, text, **kwargs):
        tag = tag[:min(6, len(tag))]
        print("[{:>6}] {}".format(tag, text))

    def log_parameters(self, params, **kwargs):
        print("<PARAMS>")
        print(params)
        print("</PARAMS>")

    def start_epoch(self, **kwargs):
        super(CLIExperimentLogger, self).start_epoch()
        print(f"\n{64 * '='}\n")
        print(f"Starting epoch {self.epoch + 1}...")

    def end_experiment(self):
        print(f"\n[  DONE] Experiment \"{self.exp_name}\" has finished.")
