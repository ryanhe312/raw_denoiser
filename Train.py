import tensorflow as tf 
import Model
import Utils

# TODO:config here

model = None
data = None

def train_step(batch):
    # TODO: complete the step
    return

def train():
    # TODO: complete all train
    return 

def test_step(batch):
    # TODO: complete the step
    return

def test():
    # TODO: complete all test
    return 

def main():
   model = Model.build_model() 
   data = Utils.get_train_batch()
   sample = data.take(1)
   result = train_step(sample)

if __name__=='__main__':
    main()
