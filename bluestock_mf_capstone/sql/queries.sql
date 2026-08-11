SELECT fund_name,aum
FROM fact_aum
ORDER BY aum DESC
LIMIT 5;

SELECT
strftime('%Y-%m',date) Month,
AVG(nav)
FROM fact_nav
GROUP BY Month;

SELECT
strftime('%Y',transaction_date),
SUM(amount)
FROM fact_transactions
WHERE transaction_type='SIP'
GROUP BY 1;

SELECT
state,
COUNT(*)
FROM fact_transactions
GROUP BY state;

SELECT *
FROM fact_performance
WHERE expense_ratio<1;