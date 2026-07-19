from abc import ABC, abstractmethod


class BaseCleaningAlgorithm(ABC):
    @abstractmethod
    def clean(self, clean_data, current_dirty_data, **args):
        
        pass
