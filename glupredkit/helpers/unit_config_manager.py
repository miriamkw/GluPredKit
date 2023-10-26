import json
import pkg_resources
import os


class UnitConfigManager:
    def __init__(self):
        self.default_config_file = pkg_resources.resource_filename('glupredkit', 'unit_config.json')

        self.user_config_dir = os.path.expanduser("~/.glupredkit")
        self.user_config_file = os.path.join(self.user_config_dir, "unit_config.json")  # Fixed the path

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
        with open(self.user_config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def convert_value(self, value):
        if not self.use_mgdl:
            return value / 18.018
        return value


# You can create a global instance of the config manager.
unit_config_manager = UnitConfigManager()
