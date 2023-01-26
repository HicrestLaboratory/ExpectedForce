
#!/bin/sh
graph=$1

#echo 
#./expected_force_serial ${graph} 1 out.txt
echo "Run CPU"
for th in 1 2 4 8 16 32 64 128
do
	echo =====================
	echo `date`
	echo threads = ${th}
	OMP_NUM_THREADS=${th} ./expected_force_openmp ${graph} 1 /tmp/out.txt
	echo =====================
	echo ""
	echo ""
done


