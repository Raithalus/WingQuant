
for i = 1:459
CoordCheck(i,:) = length(intersect(bw16{1,i}, CoordArray.B2(:,:)));
end

figure,
plot(CoordCheck)

[ccx ccy] = max(CoordCheck)