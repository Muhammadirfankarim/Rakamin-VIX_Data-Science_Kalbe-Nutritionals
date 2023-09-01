--Query 1: Berapa rata-rata umur customer jika dilihat dari marital statusnya?
SELECT 
    "Marital_Status", 
    AVG("Age") AS average_age
FROM 
    "casestudy_customer"
GROUP BY 
    "Marital_Status";

--Query 2: Berapa rata-rata umur customer jika dilihat dari gender nya?
SELECT 
    CASE 
        WHEN "Gender" = 1 THEN 'Male'
        WHEN "Gender" = 0 THEN 'Female'
        ELSE 'Other'
    END AS Gender_Type, 
    AVG("Age") AS average_age
FROM 
    "casestudy_customer"
GROUP BY 
    Gender_Type;

-- Query 3: Tentukan nama store dengan total quantity terbanyak!
SELECT
    s.storename,
    SUM(t.qty) AS total_quantity
FROM
    casestudy_store s
JOIN
    casestudy_transaction t ON s.storeid = t.storeid
GROUP BY
    s.storename
ORDER BY
    total_quantity DESC
LIMIT 1;

-- Query 4: Tentukan nama produk terlaris dengan total amount terbanyak!
SELECT
    p.product_name,
    SUM(t.totalamount) AS total_amount
FROM
    casestudy_product p
JOIN
    casestudy_transaction t ON p.productid = t.productid
GROUP BY
    p.product_name
ORDER BY
    total_amount DESC
LIMIT 1;