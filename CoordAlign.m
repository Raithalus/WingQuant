
for i = 1:459
CoordCheck(i,:) = length(intersect(bw16{1,i}, CoordArray.Q4(:,:)));
end

figure,
plot(CoordCheck)

[ccx ccy] = max(CoordCheck)