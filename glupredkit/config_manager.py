import json
import pkg_resources
import os


class ConfigManager:
    def __init__(self):
        self.config_file = pkg_resources.resource_filename('glupredkit', 'config.json')
        with open(self.config_file) as f:
            self.config = json.load(f)

        self.default_config_file = pkg_resources.resource_filename('glupredkit', 'config.json')
        self.user_config_dir = os.path.expanduser("~/.glupredkit")
        self.user_config_file = os.path.join(self.user_config_dir, "config.json")

        if not os.path.exists(self.user_config_file):
            self._initialize_user_config()

        with open(self.user_config_file, 'r') as f:
            self.config = json.load(f)

    def _initialize_user_config(self):
        os.makedirs(self.user_config_dir, exist_ok=True)
        with open(self.default_config_file, 'r') as src, open(self.user_config_file, 'w') as dst:
            dst.write(src.read())


    @property
    def use_mgdl(self):
        return self.config.get('use_mgdl', True)

    @use_mgdl.setter
    def use_mgdl(self, value):
        self.config['use_mgdl'] = value
        # Update the config.json file to persist the change
        with open(self.user_config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def convert_value(self, value):
        if not self.use_mgdl:
            return value / 18.018
        return value


# You can create a global instance of the config manager.
config_manager = ConfigManager()
