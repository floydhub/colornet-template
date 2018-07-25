# Colornet

Today, colorization is done by hand in Photoshop, a picture can take up to one month to colorize. It requires extensive research. A face alone needs up to 20 layers of pink, green and blue shades to get it just right. But something changed this year when Amir Avni used neural networks to [troll the subreddit](http://www.whatimade.today/our-frst-reddit-bot-coloring-b-2/) [/r/Colorization](https://www.reddit.com/r/Colorization/) - a community where people colorize historical black and white images manually using Photoshop. They were astonished with Amir’s deep learning bot - what could take up to a month of manual labour could now be done in just a few seconds.

### Try it now

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/floydhub/colornet-template)

Click this button to open a Workspace on FloydHub that will train this model.

### Colorizing Black&White photos

Fascinated by Amir’s neural network, Emill reproduced it and documented the process in the famous blog post: [Colorizing B&W Photos with Neural Networks](https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/). In this notebook we will reproduce Emil's work by using the Full Version of his experiments.

![colorization](https://blog.floydhub.com/content/images/2018/06/woman_results-1-min.png)
*The middle picture is done with our neural network and the picture to the right is the original color photo - Image from the [Blog](https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/)*

We will:
- Preprocess the image data for this CV task
- Build and train the `colornet` model using Keras and Tensorflow
- Evaluate our model on the test set
- Run the model on your own black&white and colored pictures!


### Serve an interactive web page for your own model

You can easily spin up a serve job on FloydHub to demo your model through an
interactive web site. Just run the following command from workspace terminal or
your local machine:

```bash
floyd run --mode serve
```

You should be able to see the following page when visiting the FloydHub serve url:

![serve-example](https://static.floydhub.com/images/color-serve-example.png)
