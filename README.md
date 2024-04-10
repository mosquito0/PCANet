# PCANet
Dynamic pricing, which suggests the optimal prices based on the
dynamic demands, has received considerable attention in academia
and industry. On online hotel booking platforms, room demand
fluctuates due to various factors, notably hotel popularity and competition.
In this paper, we propose a dynamic pricing approach with
popularity and competitiveness-aware demand learning. Specifically,
we introduce a novel demand function that incorporates
popularity and competitiveness coefficients to comprehensively
model the price elasticity of demand. We develop a dynamic demand
prediction network that focuses on learning these coefficients
in the proposed demand function, enhancing the interpretability
and accuracy of price suggestion. The model is trained in a multitask
framework that effectively leverages the correlations of demands
among groups of similar hotels to alleviate data sparseness
in room-level occupancy prediction. Comprehensive experiments
are conducted on real-world datasets, which validate the superiority
of our method over the state-of-the-art baselines for both
demand prediction and dynamic pricing. Our model has been successfully
deployed on a popular online travel platform, serving tens
of millions of users and hoteliers.
