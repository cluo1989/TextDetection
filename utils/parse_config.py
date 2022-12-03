# coding: utf-8
import os
import yaml
from argparse import ArgumentParser



class ArgsParser(ArgumentParser):
    '''Arguments parser class

    parse args from command line and load from config file.
    '''
    def __init__(self):
        super(ArgsParser, self).__init__()
        self.add_argument("-c", "--config", required=True, help="config file to use")
        self.add_argument("-o", "--option", nargs='+', help="optional configs")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.option = self._parse_opt(args.option)
        args.config = self._load_config(args.config)

        # TODO: merge option and config
        return args.config

    def _load_config(self, file_path):
        """
        Load config from yml/yaml file.
        Args:
            file_path (str): Path of the config file to be loaded.
        Returns: config
        """
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        cfgs = yaml.load(open(file_path), Loader=yaml.Loader)
        return cfgs

    def _parse_opt(self, opts):
        """
        Parse optional args.

        """
        config = {}
        if not opts:
            return config
        
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config
