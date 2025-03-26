Generating Subscriber Data
================
Tucker Morgan
2025-03-25

## Introduction

In this project, I’ll be showing a few ways to train a subscriber
retention scoring model; the goal of this model will be to take
historical subscriber-level data points as input and predict the
likelihood of subscription renewal in the next subscription period. For
this project, let’s consider YouTube Premium, [which passed 125M
subscribers in
2025](https://variety.com/2025/digital/news/youtube-125-million-music-premium-subscribers-lite-tier-1236328177/),
as our subscription service.

First, we will need some data… and since data on YouTube Premium
subscriptions is publicly available, we will use this document to
generate synthetic data for model training. This isn’t ideal, but the
focus of this project is to walk through the model training process not
to draw conclusions about Google’s business as it relates to YouTube
Premium. To generate synthetic data, we will use a handful of assumed
distributions for various user-level metrics, and we will need to
provide some true relationship, $f$, between our generated metrics and
our outcome of interest.

## Generating Metrics

While there are over 125M subscribers to YouTube Premium, in an effort
to reduce the overhead of this task and to make the size of data more
manageable on local machines, let’s start with 1.5M subscribers. And we
will start the following metrics:

- `subscription_tenure` - the number of consecutive subscription
  periods, call it a month, for which this customer has been a
  subscriber
- `total_user_sessions` - the total number of distinct sessions a user
  had in the most recent month
- `total_duration_min` - the total duration of a user’s sessions in the
  most recent month (in minutes)
- `total_watch_duration_mins` - the total duration of a user’s watch
  time in the period (in minutes)
- `total_videos_watched` - the total number of videos watched by a user
  in the most recent month (not unique videos)
- `num_user_subscriptions` - the total number of channels to which the
  user is subscribed
- `num_liked_videos` - the total number of videos that a user “liked” in
  the period
- `num_comments` - the total number of comments a user posted in the
  period
- `youtube_tv_subscriber` - an indicator for if someone has YouTube
  Premium as part of their YouTube TV subscription

In actual applications, we will likely be building off of a database
with much more granular data that we would roll-up to the user x
subscription period level. I might show how to do this later, but it can
vary based on how a customer database is set up.

``` r
set.seed(100)
subscriber_df <- 
  tibble(
    user_id = seq(from = 1000001, to = 2500000)
  ) %>% 
  # for subscription tenure, we will assume a normal distribution with mean 12 months; sd of 8 months
  # however, we want the distribution truncated to be above 0 and below 72 (6 years), so it ends up looking not exactly normal
  add_column(
    subscription_tenure = ceiling(msm::rtnorm(nrow(.), mean = 12, sd = 8, lower = .01, upper = 72)),
    # we'll generate total user sessions such that they are independent of subscription tenure
    total_user_sessions = ceiling(msm::rtnorm(nrow(.), mean = 40, sd = 20, lower = -1, upper = 72))
    ) %>% 
  # now, we'll generate some user engagement metrics as functions of the above
  mutate(
    total_duration_min = total_user_sessions * abs(rnorm(nrow(.), mean = 2, sd = 10)),
    total_watch_duration_min = total_duration_min * runif(nrow(.), min = 0.6, max = 0.95),
    total_videos_watched = total_watch_duration_min / abs(rnorm(nrow(.), mean = 3, sd = 10)),
    # here we calculate the percentile of each user based on total user sessions; we then use that value to generate a number of subscriptions
    percentile = percent_rank(total_user_sessions),
    num_user_subscriptions = floor(abs((percentile * 100) + runif(nrow(.), min = -20, max = 0))),
    # and we'll make the number of likes and comments a function of user subscriptions
    num_liked_videos = ceiling(runif(nrow(.), min = 0, max = 1) * total_videos_watched),
    num_comments = round(rexp(nrow(.), rate = 2)) * num_user_subscriptions,
    # and now, i'll just make youtube TV subscriber a weighted coin flip with ~ 40% being YouTube TV
    youtube_tv_subscriber = rbinom(nrow(.), size = 1, prob = 0.3878)
  )

skimr::skim(subscriber_df)
```

|                                                  |               |
|:-------------------------------------------------|:--------------|
| Name                                             | subscriber_df |
| Number of rows                                   | 1500000       |
| Number of columns                                | 11            |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |               |
| Column type frequency:                           |               |
| numeric                                          | 11            |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |               |
| Group variables                                  | None          |

Data summary

**Variable type: numeric**

| skim_variable | n_missing | complete_rate | mean | sd | p0 | p25 | p50 | p75 | p100 | hist |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| user_id | 0 | 1 | 1750000.50 | 433012.85 | 1000001 | 1375000.75 | 1750000.50 | 2125000.25 | 2500000.00 | ▇▇▇▇▇ |
| subscription_tenure | 0 | 1 | 13.62 | 7.03 | 1 | 8.00 | 13.00 | 18.00 | 53.00 | ▇▇▂▁▁ |
| total_user_sessions | 0 | 1 | 39.13 | 16.69 | 0 | 27.00 | 40.00 | 52.00 | 72.00 | ▂▅▇▇▃ |
| total_duration_min | 0 | 1 | 318.69 | 294.94 | 0 | 97.25 | 234.06 | 454.08 | 3083.72 | ▇▁▁▁▁ |
| total_watch_duration_min | 0 | 1 | 246.93 | 232.71 | 0 | 74.21 | 179.40 | 349.73 | 2665.21 | ▇▁▁▁▁ |
| total_videos_watched | 0 | 1 | 232.88 | 14415.88 | 0 | 9.64 | 26.49 | 69.34 | 11227678.39 | ▇▁▁▁▁ |
| percentile | 0 | 1 | 0.49 | 0.29 | 0 | 0.24 | 0.50 | 0.74 | 0.99 | ▇▇▇▇▇ |
| num_user_subscriptions | 0 | 1 | 40.11 | 27.42 | 0 | 14.00 | 38.00 | 64.00 | 99.00 | ▇▅▅▅▂ |
| num_liked_videos | 0 | 1 | 119.90 | 8576.75 | 0 | 4.00 | 11.00 | 33.00 | 5999608.00 | ▇▁▁▁▁ |
| num_comments | 0 | 1 | 17.05 | 32.02 | 0 | 0.00 | 0.00 | 22.00 | 492.00 | ▇▁▁▁▁ |
| youtube_tv_subscriber | 0 | 1 | 0.39 | 0.49 | 0 | 0.00 | 0.00 | 1.00 | 1.00 | ▇▁▁▁▅ |

And now, we need to generate our outcomes. We could randomly generate
them, but I’d like to have some level of structure to outcomes so we
have something to work toward.

``` r
# this is setting our intercept,
# such that anyone with zero for all inputs has a 20% probability of renewing
p_0 <- 0.2
b_0 <- log(p_0/(1-p_0))
# we will now set the coefficients for four variables
# b_1 for subscription tenure
# b_2 for total videos watched
# b_3 for number of user subscriptions
# and b_4 for number of liked videos
# we'll see how our statistical learning methods do in identifying these variables
or_1 <- 1.102 # setting the odds ratio; each additional month of tenure increases the likelihood of renewing
b_1 <- log(or_1)
or_2 <- 1.0035 # same for total videos watched but 1 video < 1 month
b_2 <- log(or_2)
or_3 <- 1.003 # 1 subscription is slightly less influential than 1 video watched
b_3 <- log(or_3)
or_4 <- 1.0045 # 1 video liked is more influential than 1 video watched
b_4 <- log(or_4)

subscriber_df_outcomes <- 
  subscriber_df %>% 
  mutate(renew_prob = 1/(1+exp(-(b_0 + b_1 * subscription_tenure + b_2 * total_videos_watched + 
                                   b_3 * num_user_subscriptions + b_4 * num_liked_videos))),
         renew_flag = ifelse(renew_prob >= 0.5, TRUE, FALSE))
```

Let’s see what our outcomes end up looking like.

``` r
ggplot(subscriber_df_outcomes,
       aes(x = renew_prob)) +
  geom_histogram(bins = 10, fill = 'cornflowerblue', alpha = 0.6, color = 'black') +
  labs(title = 'Distribution of Renewal Probabilities', x = 'Renewal Probability', y = 'Count') +
  theme_minimal()
```

![](generating_data_files/figure-gfm/summary%20of%20outcomes-1.png)<!-- -->

``` r
subscriber_df_outcomes %>%
  group_by(renew_flag) %>% 
  summarise(n_perc = n()/nrow(.)) %>% 
  mutate(n_perc = round(n_perc, 2)) %>% 
  knitr::kable()
```

| renew_flag | n_perc |
|:-----------|-------:|
| FALSE      |   0.35 |
| TRUE       |   0.65 |

The `renewal_flag` field is what we’ll use in our analysis, and we can
see that 65% of users renew. While this may not be true to real data,
this will give us a fairly balanced population of outcomes. So let’s
remove a couple of fields and write this to a file.

``` r
export_df <- 
  subscriber_df_outcomes %>% 
  select(-percentile, -renew_prob)

# i'll use an rds file to save a little storage space
save(export_df, file = './subscribers.Rds')
#write_csv(export_df, './subscribers.csv')
```
