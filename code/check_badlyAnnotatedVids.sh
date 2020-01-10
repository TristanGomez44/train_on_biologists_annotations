for annot in ../data/$1/annotations/*phases.csv
do
  csvRowNb=$(cat $annot | wc -l)
  if [ $csvRowNb -eq $2 ]
  then
    echo $annot
  fi

done
