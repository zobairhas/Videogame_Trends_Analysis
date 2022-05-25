CREATE TABLE clean (
	appid INT NOT NULL,
	name VARCHAR NOT NULL,
	release_date DATE NOT NULL,
	english BOOL NOT NULL,
	developer VARCHAR NOT NULL,
	publisher VARCHAR NOT NULL,
	platforms VARCHAR(40) NOT NULL,
	required_age INT NOT NULL,
	categories VARCHAR NOT NULL,
	genres VARCHAR NOT NULL,
	steamspy_tags VARCHAR NOT NULL,
	achievements INT NOT NULL,
	positive_ratings INT NOT NULL,
	negative_ratings INT NOT NULL,
	average_playtime INT NOT NULL,
	median_playtime INT NOT NULL,
	owners VARCHAR(100) NOT NULL,
	price FLOAT NOT NULL,
	PRIMARY KEY (appid)
);

ALTER TABLE clean
RENAME COLUMN appid TO steam_appid;

SELECT * FROM clean;

CREATE TABLE descriptions (
	steam_appid INT NOT NULL,
	detailed_description VARCHAR NOT NULL,
	about_the_game VARCHAR NOT NULL,
	short_description VARCHAR NOT NULL,
	PRIMARY KEY (steam_appid)
);

SELECT * FROM descriptions;

CREATE TABLE media (
	steam_appid INT NOT NULL,
	header_image VARCHAR NOT NULL,
	screenshots VARCHAR NOT NULL,
	background VARCHAR NOT NULL,
	movies VARCHAR NOT NULL,
	PRIMARY KEY (steam_appid)
);

SELECT * FROM media;

SELECT cl.steam_appid,
	cl.name,
	cl.price,
	cl.release_date,
	cl.developer,
	cl.publisher,
	me.header_image,
	me.background,
	de.about_the_game,
	de.short_description
INTO final
FROM media AS me
INNER JOIN clean AS cl
ON cl.steam_appid = me.steam_appid
INNER JOIN descriptions AS de
ON cl.steam_appid = de.steam_appid;

ALTER TABLE final
ADD COLUMN linear_score FLOAT,
ADD COLUMN random_forest_score FLOAT,
ADD COLUMN length_of_time INT;

SELECT * FROM final;