<h1 align="center">Training Models</h1>
***

A model's generalization error can be expressed as the sum of three very different errors:

*Bias* <br/>
> This part of generalization error is due to wrong assumptions, suc as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.

*Variance*<br/>
> This part is due to the model's excessive sensitivity to small variations in the training data. A model with many degress of freedom is likely to have high variance, and thus to overfit the training data.

*Irreducible Error*<br/>
> This part is due to noiseness in the data itself. The only way to reduce thus part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers)

Increasing a model's complexity will typically increase its variance and reduce its bias. Conversely, reducing a model's complexity increases its bias and reduces its variance. This is why it is called tradeoff.