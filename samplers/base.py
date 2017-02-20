from abc import ABCMeta, abstractmethod


class SamplerBase(metaclass=ABCMeta):

  @abstractmethod
  def seed(self, seed_value):
    pass

  # generator function
  @abstractmethod
  def sample(self, sample_size):
    pass
