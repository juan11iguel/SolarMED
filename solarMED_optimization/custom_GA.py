from pygad import GA
import cloudpickle

class MyGA(GA):

    def __getstate__(self):
        state = self.__dict__.copy()  # get the current state
        non_picklable_attrs = self.find_non_picklable_attrs()

        print(f"Non-picklable attributes: {non_picklable_attrs}")
        for attr in non_picklable_attrs:
            del state[attr]  # remove the unpickable attributes

        print(f"Picklable attributes: {state.keys()}")
        return state

    def find_non_picklable_attrs(self):
        non_picklable_attrs = []
        for attr_name, attr_value in self.__dict__.items():
            try:
                cloudpickle.dumps(attr_value)
            except Exception:
                non_picklable_attrs.append(attr_name)
        return non_picklable_attrs