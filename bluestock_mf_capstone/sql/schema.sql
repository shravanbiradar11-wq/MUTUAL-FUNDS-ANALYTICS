CREATE TABLE dim_fund(
    fund_id INTEGER PRIMARY KEY,
    amfi_code TEXT UNIQUE,
    fund_name TEXT,
    category TEXT
);

CREATE TABLE dim_date(
    date_id INTEGER PRIMARY KEY,
    full_date DATE,
    year INTEGER,
    month INTEGER,
    day INTEGER
);

CREATE TABLE fact_nav(
    nav_id INTEGER PRIMARY KEY,
    fund_id INTEGER,
    date_id INTEGER,
    nav REAL,
    FOREIGN KEY(fund_id)
    REFERENCES dim_fund(fund_id),
    FOREIGN KEY(date_id)
    REFERENCES dim_date(date_id)
);

CREATE TABLE fact_transactions(
    transaction_id INTEGER PRIMARY KEY,
    fund_id INTEGER,
    date_id INTEGER,
    amount REAL,
    transaction_type TEXT,
    FOREIGN KEY(fund_id)
    REFERENCES dim_fund(fund_id),
    FOREIGN KEY(date_id)
    REFERENCES dim_date(date_id)
);

CREATE TABLE fact_performance(
    performance_id INTEGER PRIMARY KEY,
    fund_id INTEGER,
    return_1y REAL,
    return_3y REAL,
    return_5y REAL,
    expense_ratio REAL,
    FOREIGN KEY(fund_id)
    REFERENCES dim_fund(fund_id)
);

CREATE TABLE fact_aum(
    aum_id INTEGER PRIMARY KEY,
    fund_id INTEGER,
    date_id INTEGER,
    aum REAL,
    FOREIGN KEY(fund_id)
    REFERENCES dim_fund(fund_id),
    FOREIGN KEY(date_id)
    REFERENCES dim_date(date_id)
);