# Caffeinated Pandas
## Accelerate your modeling, reporting, and development
#### 89% less RAM | 98% faster disk | 72% less storage | 48–95% faster enhancements| 10x+ faster development

This repo is a companion to a series of Medium.com articles I published in September 2021. The series' introduction can be found here: [Caffeinated Pandas: accelerate your modeling, reporting, and development](https://medium.com/@scollay/caffeinated-pandas-accelerate-your-modeling-reporting-and-development-e9d41476de3b)

## Background
I've long been interested in financial markets, trading, and investing. 

Last year, with a lot more down time and home time than expected, and armed with a few years of casual Python programming under my belt, I decided to build a backtesting and trading system that would suit MY needs. Then, if I happened on an interesting book or white paper with a trading idea, I'd have no limits to testing because I controlled the code, the market universe, and computing resources.

After months of "organic", ad-hoc data gathering and testing, I realized that I lacked even the most basic data processing capabilities and the infrastructure needed to efficiently test my trading and portfolio building ideas on a broad range of stocks and markets.

I discovered that there were four productivity killers that were really holding me back during the development and testing phases of my modeling:

1. **Running out of memory.** Solution: [Squeezing Pandas - 89% less RAM consumption](https://medium.com/@scollay/squeezing-pandas-89-less-ram-consumption-4d91a0eb9c08)
2. **Disk reading and writing taking way too long.** Solution: [Storing Pandas - 98% faster disk reads and 72% less space](https://medium.com/@scollay/storing-pandas-98-faster-disk-reads-and-72-less-space-208e2e2be8bb)
3. **Computer's full processing power not being used.** Solution: [Multiprocessing Pandas - 46 to 95% faster Dataframe enhancements](https://medium.com/@scollay/multiprocessing-pandas-46-95-faster-dataframe-enhancements-c65ef29f03b1)
4. **Taking too long to develop, iterate, and test code.** Solution: [Processing Pandas - 10x+ faster coding and iteration with "rough samples"](https://medium.com/@scollay/processing-pandas-10x-faster-coding-and-iteration-with-rough-samples-78b75b7d5b0b)

*Note:* After a few articles you may be presented with a Medium.com paywall. A 1-month subscription costs a mere $5.00 and helps to support thousands of writers.

## Not just for stocks
It should be noted that while I'm clearly focused on stock quotes in this series of articles, the principles will certainly work on any columnar, Pandas-based data for all types of models and analysis.

## Why I've written and published this
It's taken me a long time with many false starts to get to this point where I can confidently process large swaths of data. Going into this I had a notion that I'd like to write about it. Just knowing that others would be reading it, I've dramatically improved my own code and its performance, sometimes by an order of magnitude as I searched for "better" ways. Also, by sharing, I'm hoping others will provide constructive feedback on what I could have done better!


## Requirements

### Python version
On Mac I intalled with Miniconda: Python 3.9.5

On Linux/Ubuntu server: Python 3.8.2

### Libraries
*apt install python3-pip*

*pip install pandas*

*pip install psutil*

*pip install pyarrow*

*pip install scipy*

tables library: *conda install pytables* for Miniconda installation, or *pip install tables* for Ubuntu's Python

## Feedback
Please feel free to reach out to me directly at scollay@coldbrew.cc
