USE spanish_issues_db;

select * from rentas;

/*Selecionar 
-La CCAA con mayor renta en 2024
- La CCAA con menor renta en 2024
 */

SELECT Comunidades_y_ciudades_autónomas,  max(total) AS Renta
from rentas
WHERE Comunidades_y_ciudades_autónomas NOT IN ('Total Nacional')
AND periodo = 2024
group by Comunidades_y_ciudades_autónomas
order by Renta DESC
limit 1;

SELECT Comunidades_y_ciudades_autónomas,  min(total) AS Renta
from rentas
WHERE Comunidades_y_ciudades_autónomas NOT IN ('Total Nacional')
AND periodo = 2024
group by Comunidades_y_ciudades_autónomas
order by Renta ASC
limit 1;


/*
Tabla por año
*/

SELECT periodo, Comunidades_y_ciudades_autónomas, total
FROM (
    SELECT 
        periodo,
        Comunidades_y_ciudades_autónomas,
        total,
        ROW_NUMBER() OVER (PARTITION BY periodo ORDER BY total DESC) AS rn
    FROM rentas
    WHERE Comunidades_y_ciudades_autónomas <> 'Total Nacional'
) x
WHERE rn = 1;
