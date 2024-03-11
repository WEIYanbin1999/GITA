variable="Today is Aug 9, 2023"

if [[ $variable == *"Aug"* ]]; then
  echo "变量包含'Aug'"
else
  echo "变量不包含'Aug'"
fi