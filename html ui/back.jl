# Backend for number recogniser ai
# The Genie package is equavialent to Flask in python

using Genie, Genie.Router, Genie.Requests

using Flux, MLUtils, BSON
function recognizeDigit(img)
    # load the model
    BSON.@load "digits.bson" model
    # Convert image to grayscale
    img = Gray.(img)
    # Invert each pixel color
    img = (x->Gray(1)-x.val).(img)
    # resize image to 28x28 pixels
    img = imresize(img, (28, 28))
    # Get matrix of image
    digit_data = Float32.(channelview(img))
    # Flatten the image
    flat_digit_data = reshape(digit_data, (28 * 28, 1, 1, 1))
    # predict the digit (get probabilities)
    probs = model(flat_digit_data)
    # return the digit with the largest 
    # probability, converted to a string
    return "$(argmax(probs)[1]-1)"
end

route("/") do 
    return String(read("index.html"))
end

using Images
route("/api/recognize", method=POST) do
    result = ""
    files = filespayload();   
    for index in 1:10
        file = files["$index.png"]
        img = load(IOBuffer(file.data))
        result *= recognizeDigit(img)        
    end  
    @show result
    return result
end

up(8080, async=false) # run a web server on port 8080