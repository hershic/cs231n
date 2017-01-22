from abc import ABCMeta, abstractmethod


class ImporterBase(metaclass=ABCMeta):
  @abstractmethod
  def import_batch(self, batch):
    pass
