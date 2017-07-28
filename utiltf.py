
class Component(object):
    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
