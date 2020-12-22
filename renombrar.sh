#codigo para obtener la media anual de las variables usadas en las clases de clima dinamico y las variables mensuales necesarias
cd /datos/S2S/ECCC/
for f in ECMWF*.grib; do
	b="$(echo $f | sed s/ECMWF/ECCC/)"
	mv "$f" "$b"
done

