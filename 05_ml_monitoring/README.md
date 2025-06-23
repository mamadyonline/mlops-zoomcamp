### ML monitoring

1. Why ?
   1. ML models degrade over time
2. Need to monitor 
   1. service health (does it work ? Look at uptime for example)
   2. model health (is the model still relevant ? How does it perform ? Did anything break ? Look at data and concept drift) 
   3. data health (Where does the model break ? Look into data quality and integrity)
3. How to monitor ?
   1. Some tools: Prometheus, Grafana, Looker
4. Monitoring scheme
   1. software service ==> I/O logging ==> Monitoring jobs (compare reference data with ground truth) ==> ML evaluation store
   2. 