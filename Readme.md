# NPB in Rust
Implementação de alguns benchmarks da Nasa https://www.nas.nasa.gov/software/npb.html <br>

# Build and Run
Para compilar um arquivo específico existe um arquivo Makefile, é necessário possuir o cargo, o make e o compilador Rust. Para compilar o EP:
 <br>
`make ep`
<br>
Para compilar o CG:
<br>
`make cg`
<br>

Para alterar a classe do benchmark CG é preciso alterar a **linha 21** do arquivo src/cg.rs:
<br>
`let class = Class::B;`
<br>
substituindo `B` pela classe desejada.

<br>
Para alterar a classe do benchmark EP é preciso alterar a **linha 17** do arquivo src/ep.rs:
<br>

`let class = Class::A;`
<br>
substituindo `A` pela classe desejada
<br> 

Para executar, basta acessar o diretório target/release e executar o binário `cg` ou `ep`