using BSON
using Flux
using MLDatasets
using ImageShow

# Just a testing file

BSON.@load "digits.bson" model

train_x_raw, train_y_raw = MNIST(split=:train)[:];   # náum í allt trainig data frá MNIST, skipt í x og y
test_x_raw, test_y_raw = MNIST(split=:test)[:];      # náum í allt test data frá MNIST

train_x = Flux.flatten(train_x_raw); # breyta 28x28 fylkjum í vigur til að vinna með
test_x = Flux.flatten(test_x_raw);   # This results in a 784x60000 and a 784x10000-element 
                                    # array respectively.
train_y = Flux.onehotbatch(train_y_raw, 0:9);
test_y = Flux.onehotbatch(test_y_raw, 0:9);

function test_digit(num)
    return train_x_raw[:,:,num]
end

convert2image(MNIST, test_digit(10))