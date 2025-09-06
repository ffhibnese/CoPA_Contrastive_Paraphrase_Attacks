from abc import ABC, abstractmethod


class Detector(ABC):
    @abstractmethod
    def llm_likelihood(self, query: str):
        pass

    @abstractmethod
    def human_likelihood(self, query: str):
        pass
