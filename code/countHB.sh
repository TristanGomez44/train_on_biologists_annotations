for annot in ../data/$1/annotations/*phases.csv
do
  csvRowNb=$(cat $annot | grep "tHB" | wc -l)
  if [ $csvRowNb -eq 1 ]
  then
    echo $annot
  fi

done
