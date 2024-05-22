from pygad import GA
import pickle
import json
from pathlib import Path


def find_non_picklable_attrs(object):
    non_picklable_attrs = []
    for attr_name, attr_value in object.items():
        # print(f"Checking attribute: {attr_name}")

        # Manually remove some problematic attributes
        # if attr_name in ['solarMED', 'model_instance', '_matlab', 'fitness_function']:
        #     non_picklable_attrs.append(attr_name)
        #     continue

        try:
            pickle.dumps(attr_value)
        except pickle.PicklingError:
            non_picklable_attrs.append(attr_name)
        except TypeError:
            non_picklable_attrs.append(attr_name)

    print(f"Non-picklable attributes: {non_picklable_attrs}")
    return non_picklable_attrs

class MyGA(GA):

    # def __getstate__(self):
    #     state = self.__dict__.copy()  # get the current state
    #     non_picklable_attrs = find_non_picklable_attrs(state)
    #
    #     print(f"Non-picklable attributes: {non_picklable_attrs}")
    #     for attr in non_picklable_attrs:
    #         del state[attr]  # remove the unpickable attributes
    #
    #     print(f"Picklable attributes: {state.keys()}")
    #     return state



    def model_dump(self, output_path: Path | str) -> None:

        output_path = Path(output_path)
        # set file format
        output_path.with_suffix('.pkl')

        state = self.__dict__.copy()
        non_picklable_attrs = find_non_picklable_attrs(state)
        for attr in non_picklable_attrs:
            del state[attr]

        with open(output_path, 'wb') as f:
            pickle.dump(state, f)

    def model_load(self, input_path: Path | str) -> None:
        input_path = Path(input_path)
        with open(input_path, 'rb') as f:
            state = pickle.load(f)
            self.__dict__.update(state)