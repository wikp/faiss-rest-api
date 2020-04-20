#![feature(proc_macro_hygiene, decl_macro)]

use clap::{App, Arg};
use faiss::ConcurrentIndex;
use faiss::{FlatIndex};
use rocket::{State, post, routes};
use rocket_contrib::json::{Json, JsonValue};
use rocket_contrib::{json};
use serde::{Deserialize, Serialize};
struct Boot {
    index: FlatIndex
}

#[derive(Deserialize)]
struct Query {
    vectors: Vec<Vec<f32>>,
    k: usize,
}

#[derive(Serialize)]
#[derive(Debug)]
struct Neighbor {
    id: Option<u64>,
    score: f32,
}

#[derive(Serialize)]
#[derive(Debug)]
struct SingleResult {
    neighbors: Vec<Neighbor>,
    vector: Vec<f32>,
}

#[derive(Serialize)]
#[derive(Debug)]
struct Result {
    results: Vec<SingleResult>
}

#[post("/faiss/search", format = "json", data = "<query>")]
fn index(query: Json<Query>, boot: State<Boot>) -> JsonValue {
    // We have to flatten 2-dim vectors list because faiss API accepts &[f32]
    let flat_vectors: Vec<f32> = query.vectors.concat();

    if flat_vectors.len() % 64 != 0 {
        return json!({ "error":"Provided vectors has different dimensions than the index"} );
    }

    let search_result = boot.index.search(flat_vectors.as_slice(), query.k).unwrap();

    let dist_chunked = search_result.distances.chunks(query.k);
    let lab_chunked = search_result.labels.chunks(query.k);
    let mut result: Vec<SingleResult> = Vec::new();

    for it in dist_chunked.zip(lab_chunked).zip(query.vectors.iter()) {
        let (l, vec) = it;
        let (ids, scores) = l;
        let sr = SingleResult {
            neighbors: ids.iter().zip(scores)
            .filter(|(_score, id)| id.get().is_some())
            .map(|(score, id)|
                Neighbor {
                id: id.get(), //The above filter should gaurd this being none
                score: score.to_owned(),
            }).collect(),
            vector: vec.to_owned(),
        };
        result.push(sr);
    }

    return json!(result);
}

fn main() {
    let matches = App::new("Faiss REST API")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Piotr Wikieł <piotr.wikiel@gmail.com")
        .about("REST API for Faiss search operation")
        .arg(Arg::with_name("index-location")
            .short("i")
            .long("index-location")
            .value_name("INDEX_LOCATION")
            .help("Filesystem location of the index")
            .takes_value(true))
        .get_matches();

    let file_name = matches.value_of("index-location")
        .unwrap_or("/Users/piotr.wikiel/x/small_index_l2");
    println!("Using {} as an index", file_name);
    let index = faiss::read_index(&file_name).unwrap();

    rocket::ignite()
        .manage(Boot {
            index: index.into_flat().unwrap()
        })
        .mount("/", routes![index]).launch();
}
