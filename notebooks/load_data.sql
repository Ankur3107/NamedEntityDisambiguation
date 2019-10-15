.mode csv 

DROP TABLE IF EXISTS wikipages_cleaned;
DROP TABLE IF EXISTS triplets_raw;

CREATE TABLE wikipages_cleaned(
  "page_title" TEXT,
  "page_is_redirect" INT,
  "page_len" INT,
  "wikidata_numeric_id" INT,
  "views" INT,
  "page_id" INT,
  "target_page_id" INT,
  "target_page_title" TEXT
);

CREATE TABLE triplets_raw(
  "source_item_id" INT,
  "edge_property_id" INT,
  "target_item_id" INT,
  "el_rank" INT
);

.import ../data/wikipages_cleaned.csv wikipages_cleaned

.import ../data/raw/wikidata_20190805.qpq_item_statements.csv triplets_raw

-- First row is header
DELETE FROM wikipages_cleaned WHERE wikidata_numeric_id IN (SELECT wikidata_numeric_id FROM wikipages_cleaned LIMIT 1);
DELETE FROM triplets_raw WHERE source_item_id IN (SELECT source_item_id FROM triplets_raw LIMIT 1);

-- Create table of triplets of direct edges b/w Wikipages
DROP TABLE IF EXISTS wikipage_triplets;

CREATE TABLE 
    wikipage_triplets AS 
SELECT 
    T.source_item_id, 
--     W.target_page_id AS source_page_id,
    T.edge_property_id,
    T.target_item_id,
--     W2.target_page_id as target_page_id,
    T.el_rank
FROM
    (SELECT DISTINCT target_page_id, wikidata_numeric_id FROM wikipages_cleaned) as W, 
    (SELECT DISTINCT target_page_id, wikidata_numeric_id FROM wikipages_cleaned) as W2, 
    triplets_raw as T
WHERE
    W.wikidata_numeric_id = T.source_item_id
    AND W2.wikidata_numeric_id = T.target_item_id
;

DROP TABLE IF EXISTS plaintext;
DROP TABLE IF EXISTS link;
DROP TABLE IF EXISTS raw_anchor;

CREATE TABLE plaintext(
    "section_id" INT,
    "page_id" INT,
    "wikidata_numeric_id" INT,
    "section_num" INT,
    "section_name" TEXT,
    "section_text" TEXT,
    "section_len" TEXT
);

CREATE TABLE link(
    "link_id" INT,
    "source_section_id" INT,
    "source_section_num" INT,
    "source_page_id" INT,
    "source_wikidata_numeric_id" INT,
    "link_anchor" TEXT,
    "link_offset_start" INT,
    "link_offset_end" INT,
    "link_type" TEXT,
    "target_page_id" INT,
    "target_wikidata_numeric_id" INT
);

CREATE TABLE raw_anchor(
    "anchor_text" TEXT,
    "target_page_id" INT,
    "target_wikidata_numeric_id" INT,
    "anchor_target_count" INT,
    "anchor_frac" REAL,
    "target_frac" REAL
);

.import ../data/raw/enwiki_20190801.k_plaintext.csv plaintext
.import ../data/raw/enwiki_20190801.k_link.csv link
.import ../data/raw/enwiki_20190801.k_raw_anchors.csv raw_anchor

-- First row is header
DELETE FROM plaintext WHERE section_name IN (SELECT section_name FROM plaintext LIMIT 1);
DELETE FROM link WHERE link_anchor IN (SELECT link_anchor FROM link LIMIT 1);
DELETE FROM raw_anchor WHERE anchor_text IN (SELECT anchor_text FROM raw_anchor LIMIT 1);