function locs_new=adjust_elecs(names,locs);

[nchan ndum]=size(locs);
elec_suf=zeros(nchan,1);
locs_new=locs;
for i=1:nchan;
    nloc=names{i};
    n=length(nloc);
    if strcmp(nloc(n),'z');
        locs_new(i,1)=0;
        %disp(nloc)
        %[locs(i,:),locs_new(i,:)]
    else
        kont=0;
        while kont==0
            for k=20:-1:10;
                if strcmp(nloc(n-1:n),num2str(k));
                    elec_pre{i}=nloc(1:n-2);
                    elec_suf(i)=eval(nloc(n-1:n));
                    kont=1;
                end
            end
            for k=9:-1:1;
                if strcmp(nloc(n),num2str(k));
                    elec_pre{i}=nloc(1:n-1);
                    elec_suf(i)=eval(nloc(n));
                    kont=1;
                end
            end
        end
    end
end

for i=1:nchan
    for j=i+1:nchan
        if strcmp(elec_pre{i},elec_pre{j})
            n1=elec_suf(i);
            n2=elec_suf(j);
            if round(n2/2)*2==n2 & n1+1==n2
                x=(locs(i,1)-locs(j,1))/2;
                y=(locs(i,2)+locs(j,2))/2;
                locs_new(i,1)=x;
                locs_new(j,1)=-x;
                locs_new(i,2)=y;
                locs_new(j,2)=y;
            elseif round(n1/1)*2==n1 & n2+1==n1
                x=-(locs(i,1)-locs(j,1))/2;
                y=(locs(i,2)+locs(j,2))/2;
                locs_new(i,1)=-x;
                locs_new(j,1)=x;
                locs_new(i,2)=y;
                locs_new(j,2)=y;
            end 
        end
    end
end

     


return;

