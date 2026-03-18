# numeric (11 columns)
priority
cpus_req
mem_req (imputer by median)
time_limit (imputer by median)
time_submit x3 (hour/dayofweek/month) x2 (sin/cos) = 6 columns
eligible_delay = time_eligible - time_submit
# category (one-hot encoding) (20 columns)
partition (5 types)
constraint (7 types)
flag (4 types)
job_type (4 types)
