
#!/bin/sh
graph=$1

#echo 
#./expected_force_serial ${graph} 1 out.txt
echo "Run GPU"
for b in 64 128 256 512
do
	for th in  512
	do
		for sc in 1 
		do
			echo =====================
			echo `date`
			echo bs = ${b} threads = ${th} streams = ${sc}
			./expected_force ${graph} ${b} ${th} ${sc} 1
			echo =====================
			echo ""
			echo ""
		done
	done
done


