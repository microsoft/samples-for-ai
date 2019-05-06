import numpy as np


def get_sample(batch_size=128, in_bits=10, out_bits=8, max_size=20, min_size=1):
    # in order to be a generator, we start with an endless loop:
    while True:
        # generate samples with random length.
        # there a two flags, one for the beginning of the sequence 
        # (only second to last bit is one)
        # and one for the end of the sequence (only last bit is one)
        # every other time those are always zero.
        # therefore the length of the generated sample is:
        # 1 + actual_sequence_length + 1 + actual_sequence_length
        
        # make flags
        begin_flag = np.zeros((1, in_bits))
        begin_flag[0, in_bits-2] = 1
        end_flag = np.zeros((1, in_bits))
        end_flag[0, in_bits-1] = 1
    
        # initialize arrays: for processing, every sequence must be of the same length.
        # We pad with zeros.
        temporal_length = max_size*2 + 2
        # "Nothing" on our band is represented by 0.5 to prevent immense bias towards 0 or 1.
        inp = np.ones((batch_size, temporal_length, in_bits))*0.5
        out = np.ones((batch_size, temporal_length, out_bits))*0.5
        # sample weights: in order to make recalling the sequence much more important than having everything set to 0
        # before and after, we construct a weights vector with 1 where the sequence should be recalled, and small values
        # anywhere else.
        sw  = np.ones((batch_size, temporal_length))*0.01
    
        # make actual sequence
        for i in range(batch_size):
            ts = np.random.randint(low=min_size, high=max_size+1)
            actual_sequence = np.random.uniform(size=(ts, out_bits)) > 0.5
            output_sequence = np.concatenate((np.ones((ts+2, out_bits))*0.5, actual_sequence), axis=0)
    
            # pad with zeros where only the flags should be one
            padded_sequence = np.concatenate((actual_sequence, np.zeros((ts, 2))), axis=1)
            input_sequence = np.concatenate((begin_flag, padded_sequence, end_flag), axis=0)
            
    
            # this embedds them, padding with the neutral value 0.5 automatically
            inp[i, :input_sequence.shape[0]] = input_sequence
            out[i, :output_sequence.shape[0]] = output_sequence
            sw[i, ts+2 : ts+2+ts] = 1
    
        yield inp, out, sw
