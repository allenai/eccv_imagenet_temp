local example = "an/example/string"
for i in string.gmatch(example, "([^/]+)") do
  print(i)
end