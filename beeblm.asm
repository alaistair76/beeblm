\ BeebLM - a tiny neural language model for the BBC Micro
\ BPLUS = 0: one file, main memory only, runs on a Model B or B+ with DFS
\ BPLUS = 1: also uses the B+'s 12K of paged RAM. Three files: BEEBLM (loader),
\            W2 (hidden-layer weights, copied to &8000-&AFFF) and MAIN (this program)
\ Assemble with beebasm. Model constants and weights come from build/ (see export.py).
\
\ All arithmetic is integer. model.py holds the reference this code matches bit for bit.
\
\ Question encoder (runs once per line typed)
\   hashed character trigrams -> HQ units, 4-bit weights, 16-bit accumulate, shift, clamp 0..127
\   -> H1 units, ternary weights, shift, clamp -QC..QC, plus a "mood" row picked from the clock
\ Generator (runs once per word produced)
\   + ternary rows for the two previous words and the position -> ReLU      (8-bit accumulators)
\   -> H2 units, ternary, 16-bit accumulate, shift, clamp 0..127
\   -> N1 + N2 outputs, ternary; the largest in each group gives the two digits of the word id
\ Ternary weights are packed five to a byte (3^5 = 243).

INCLUDE "build/consts.asm"

OSASCI = &FFE3
OSNEWL = &FFE7
OSWRCH = &FFEE
OSWORD = &FFF1
OSBYTE = &FFF4
OSCLI  = &FFF7
ROMSEL = &FE30          \ bit 7 pages the B+'s 12K RAM in at &8000-&AFFF
ROMCPY = &F4            \ the OS's copy of ROMSEL

ROW_POS   = K * CTXW
ROW_MOOD  = K * CTXW + P
NPUNCT    = 4

\ ---------------------------------------------------------------- zero page
ORG &70
.ptr     SKIP 2      \ current sparse row
.acc16   SKIP 2      \ value being shifted and clamped
.vin     SKIP 2      \ dense layer input vector
.dotn    SKIP 1      \ dense layer: input count
.idx     SKIP 1      \ general loop counter
.kbyte   SKIP 1      \ byte index within a row
.pos     SKIP 1      \ words produced so far
.ctxd    SKIP 2 * K  \ digits of the last K word ids
.best    SKIP 2
.besti   SKIP 1
.d1      SKIP 1
.tok     SKIP 2
.wp      SKIP 2
.nlen    SKIP 1      \ normalised input length
.ma      SKIP 2      \ scratch
.mb      SKIP 1
.val     SKIP 1      \ dense layer: current input value
.nout    SKIP 1      \ dense layer: output count
.rowb    SKIP 1      \ dense layer: bytes of weights per input
.oldrom  SKIP 1

\ ---------------------------------------------------------------- code
ORG &1900
GUARD &7C00

.start
IF BPLUS
    lda ROMCPY          \ page the 12K RAM in for as long as we run
    sta oldrom
    ora #&80
    sta ROMCPY
    sta ROMSEL
ENDIF
    ldx #0
.banner_loop
    lda banner,x
    beq main_loop
    jsr OSASCI
    inx
    bne banner_loop

.main_loop
    lda #'>'
    jsr OSWRCH
    lda #' '
    jsr OSWRCH
    lda #0
    ldx #LO(osw_block)
    ldy #HI(osw_block)
    jsr OSWORD
    bcc got_line
    lda #&7E            \ acknowledge Escape, then restart BASIC cleanly
    jsr OSBYTE
    jsr OSNEWL
IF BPLUS
    lda oldrom
    sta ROMCPY
    sta ROMSEL
ENDIF
    ldx #LO(cmd_basic)
    ldy #HI(cmd_basic)
    jmp OSCLI

.got_line
    jsr normalise
    lda nlen
    cmp #3
    bcc main_loop       \ nothing usable typed
    jsr encode_query
    jsr generate
    jsr OSNEWL
    jmp main_loop

\ ---- normalise: linebuf (CR terminated) -> nbuf codes, space=0 a-z=1..26 0-9=27..36
\      result is " words separated by single spaces " ; nlen = length
.normalise
{
    lda #0
    sta nbuf
    ldx #1              \ write index
    ldy #0              \ read index
.loop
    lda linebuf,y
    iny
    cmp #13
    beq done
    cmp #39             \ apostrophes vanish: what's -> whats
    beq loop
    cmp #'0'
    bcc space
    cmp #'9'+1
    bcc digit
    and #&DF            \ fold to upper case
    cmp #'A'
    bcc space
    cmp #'Z'+1
    bcs space
    sbc #'A'-2          \ carry is clear, so this subtracts 'A'-1
    bne store           \ always
.digit
    sbc #'0'-28         \ carry is clear: A - ('0'-27)
    bne store           \ always
.space
    lda nbuf-1,x
    beq loop            \ collapse runs of spaces
    lda #0
.store
    sta nbuf,x
    inx
    bne loop
.done
    lda nbuf-1,x
    beq fin
    lda #0
    sta nbuf,x
    inx
.fin
    stx nlen
    rts
}

\ ---- encode_query: question -> qacc
.encode_query
{
    ldx #0
    txa
.clr
    sta qflag,x
    inx
IF Q < 256
    cpx #Q
ENDIF
    bne clr

    ldx #0              \ two hashes for every trigram
.tri
    ldy nbuf,x
    lda ht0,y
    sta ma
    lda ht1,y
    sta ma+1
    ldy nbuf+1,x
    lda ht1,y
    eor ma
    sta ma
    lda ht2,y
    eor ma+1
    sta ma+1
    ldy nbuf+2,x
    lda ht2,y
    eor ma
    and #Q-1
    tay
    lda #1
    sta qflag,y
    ldy nbuf+2,x
    lda ht0,y
    eor ma+1
    and #Q-1
    tay
    lda #1
    sta qflag,y
    inx
    inx
    inx
    cpx nlen            \ carry clear while another trigram remains
    dex
    dex
    bcc tri

    ldx #HQ             \ first layer: 16-bit accumulators start at the biases
.bias
    lda bqlo-1,x
    sta alo-1,x
    lda bqhi-1,x
    sta ahi-1,x
    dex
    bne bias

    lda #LO(wq)
    sta ptr
    lda #HI(wq)
    sta ptr+1
    lda #0
    sta idx
.rows
    ldy idx
    lda qflag,y
    beq next
    jsr add_qrow
.next
    clc
    lda ptr
    adc #ROWBQ
    sta ptr
    bcc nc
    inc ptr+1
.nc
    inc idx
IF Q < 256
    lda idx
    cmp #Q
ENDIF
    bne rows

    ldx #0              \ shift and clamp into zv
.act
    lda alo,x
    sta acc16
    lda ahi,x
    sta acc16+1
    ldy #SQ
    jsr shift_clamp
    sta zv,x
    inx
    cpx #HQ
    bne act

    \ second layer: ternary, HQ -> H1, result shifted and clamped into acc
    ldx #H1
.zb
    lda bzlo-1,x
    sta alo-1,x
    lda bzhi-1,x
    sta ahi-1,x
    dex
    bne zb
    lda #LO(zv)
    sta vin
    lda #HI(zv)
    sta vin+1
    lda #HQ
    sta dotn
    lda #LO(wz)
    sta ptr
    lda #HI(wz)
    sta ptr+1
    lda #H1
    sta nout
    lda #H1 DIV 5
    sta rowb
    jsr layer
    ldx #0
.l2
    lda alo,x
    sta acc16
    lda ahi,x
    sta acc16+1
FOR i, 1, SZ            \ arithmetic shift right
    lda acc16+1
    cmp #&80
    ror acc16+1
    ror acc16
NEXT
    lda acc16+1
    bmi neg
    bne satp
    lda acc16
    cmp #QC+1
    bcc store
.satp
    lda #QC
    bne store
.neg
    cmp #&FF
    bne satn
    lda acc16
    cmp #256-QC
    bcs store
.satn
    lda #256-QC
.store
    sta acc,x
    inx
    cpx #H1
    bne l2

    lda #1              \ mood from the clock: picks between reply variants
    ldx #LO(clock)
    ldy #HI(clock)
    jsr OSWORD
    lda clock
    and #M-1
    clc
    adc #ROW_MOOD
    jsr set_row
    jsr add_row

    ldx #H1
.save
    lda acc-1,x
    sta qacc-1,x
    dex
    bne save
    rts
}

\ ---- generate: produce and print the reply one word at a time
.generate
{
    lda #0
    sta pos
    ldx #2*K-1
.init
    sta ctxd,x
    dex
    bpl init

.next_tok
    ldx #H1
.copy
    lda qacc-1,x
    sta acc-1,x
    dex
    bne copy

    lda #0
    sta idx
.ctx_rows
    ldx idx
    lda ctxd,x
    clc
    adc ctx_base,x
    jsr set_row
    jsr add_row
    inc idx
    lda idx
    cmp #2*K
    bne ctx_rows

    lda pos
    clc
    adc #ROW_POS
    jsr set_row
    jsr add_row

    ldx #H1             \ ReLU
.relu
    lda acc-1,x
    bpl ok
    lda #0
    sta acc-1,x
.ok
    dex
    bne relu

    \ ---- hidden layer: ternary, H1 -> H2
    ldx #H2
.hb
    lda b2lo-1,x
    sta alo-1,x
    lda b2hi-1,x
    sta ahi-1,x
    dex
    bne hb
    lda #LO(acc)
    sta vin
    lda #HI(acc)
    sta vin+1
    lda #H1
    sta dotn
    lda #LO(w2)
    sta ptr
    lda #HI(w2)
    sta ptr+1
    lda #H2
    sta nout
    lda #H2 DIV 5
    sta rowb
    jsr layer
    ldx #0
.l2
    lda alo,x
    sta acc16
    lda ahi,x
    sta acc16+1
    ldy #S2
    jsr shift_clamp
    sta h2,x
    inx
    cpx #H2
    bne l2

    \ ---- output layer: best of the first N1, then best of the next N2
    ldx #NOUT
.ob
    lda b3lo-1,x
    sta alo-1,x
    lda b3hi-1,x
    sta ahi-1,x
    dex
    bne ob
    lda #LO(h2)
    sta vin
    lda #HI(h2)
    sta vin+1
    lda #H2
    sta dotn
    lda #LO(w3)
    sta ptr
    lda #HI(w3)
    sta ptr+1
    lda #NOUT
    sta nout
    lda #NOUT DIV 5
    sta rowb
    jsr layer
    lda #0
    sta idx
.l3
    ldx idx
    beq take
    cpx #N1
    bne compare
    lda besti           \ first group finished
    sta d1
    jmp take
.compare
    lda best            \ signed compare: is best < this output ?
    cmp alo,x
    lda best+1
    sbc ahi,x
    bvc nv
    eor #&80
.nv
    bpl keep
.take
    lda alo,x
    sta best
    lda ahi,x
    sta best+1
    stx besti
.keep
    inc idx
    lda idx
    cmp #NOUT
    bne l3

    sec                 \ word id = d1 * N2 + d2
    lda besti
    sbc #N1
    sta besti
    sta tok
    lda #0
    sta tok+1
    ldx d1
    beq got_tok
.mul
    clc
    lda tok
    adc #N2
    sta tok
    bcc mnc
    inc tok+1
.mnc
    dex
    bne mul
.got_tok
    lda tok
    ora tok+1
    beq done            \ id 0 ends the reply
    lda tok
    cmp #LO(NV)
    lda tok+1
    sbc #HI(NV)
    bcs done            \ not a real word: stop
    jsr print_tok

FOR i, 0, 2*K-3
    lda ctxd+i+2
    sta ctxd+i
NEXT
    lda d1
    sta ctxd+2*K-2
    lda besti
    sta ctxd+2*K-1
    inc pos
    lda pos
    cmp #MAXTOK
    beq done
    jmp next_tok
.done
    rts
}

\ ---- print_tok: ids 1..4 are punctuation, the rest index the word list
.print_tok
{
    lda tok+1
    bne word
    ldx tok
    cpx #NPUNCT+1
    bcs word
    lda punct-1,x
    jmp OSWRCH
.word
    lda pos
    beq nospace
    lda #' '
    jsr OSWRCH
.nospace
    lda #LO(words)
    sta wp
    lda #HI(words)
    sta wp+1
    sec
    lda tok
    sbc #NPUNCT+1
    sta ma
    lda tok+1
    sbc #0
    sta ma+1
    ldy #0
.skip
    lda ma
    ora ma+1
    beq show
.skipch
    lda (wp),y
    inc wp
    bne s1
    inc wp+1
.s1
    asl a
    bcc skipch          \ last letter of a word has its top bit set
    lda ma
    bne s2
    dec ma+1
.s2
    dec ma
    jmp skip
.show
    lda (wp),y
    pha
    and #&7F
    jsr OSWRCH
    inc wp
    bne s3
    inc wp+1
.s3
    ldy #0
    pla
    bpl show
    rts
}

\ ---- shift_clamp: A = clamp(acc16 >> Y, 0, 127)     (Y >= 1; keeps X)
.shift_clamp
{
    lda acc16+1
    bmi zero
.sh
    lsr acc16+1
    ror acc16
    dey
    bne sh
    lda acc16+1
    bne sat
    lda acc16
    bpl done
.sat
    lda #127
    rts
.zero
    lda #0
.done
    rts
}

\ ---- add_qrow: aq[0..HQ-1] += row of 4-bit weights at ptr (stored as value+8)
.add_qrow
{
    ldy #0
    ldx #0
.loop
    lda (ptr),y
    pha
    and #15
    jsr addv
    pla
    lsr a
    lsr a
    lsr a
    lsr a
    jsr addv
    iny
    cpy #ROWBQ
    bne loop
    rts
.addv
    sec
    sbc #8
    beq done
    bmi minus
    clc
    adc alo,x
    sta alo,x
    bcc done
    inc ahi,x
    bcs done
.minus
    clc
    adc alo,x
    sta alo,x
    bcs done
    dec ahi,x
.done
    inx
    rts
}

\ ---- set_row: ptr = w1 + A * ROWB1
.set_row
{
    sta ma
    lda #ROWB1
    sta mb
    lda #0
    sta ptr
    sta ptr+1
    ldy #8
.loop
    asl ptr
    rol ptr+1
    asl mb
    bcc skip
    clc
    lda ptr
    adc ma
    sta ptr
    bcc skip
    inc ptr+1
.skip
    dey
    bne loop
    clc
    lda ptr
    adc #LO(w1)
    sta ptr
    lda ptr+1
    adc #HI(w1)
    sta ptr+1
    rts
}

\ ---- add_row: acc[0..H1-1] += CR * ternary row at ptr
MACRO L1TRIT table
{
    lda table,y
    beq done
    bmi minus
    clc
    lda acc,x
    adc #CR
    sta acc,x
    jmp done
.minus
    sec
    lda acc,x
    sbc #CR
    sta acc,x
.done
    inx
}
ENDMACRO

.add_row
{
    lda #0
    sta kbyte
    tax
.loop
    ldy kbyte
    lda (ptr),y
    tay
    L1TRIT tr0
    L1TRIT tr1
    L1TRIT tr2
    L1TRIT tr3
    L1TRIT tr4
    inc kbyte
    cpx #H1
    beq done
    jmp loop
.done
    rts
}

\ ---- layer: a dense ternary layer, done one INPUT at a time so zero inputs cost nothing
\      alo/ahi[0..nout-1] += sum over inputs i of vin[i] * weights[i][..]   (weights at ptr, rowb bytes per input)
.layer
{
    lda #0
    sta idx
.loop
    ldy idx
    lda (vin),y
    beq skip            \ most activations are zero after ReLU
    sta val
    jsr add_col
.skip
    clc
    lda ptr
    adc rowb
    sta ptr
    bcc nc
    inc ptr+1
.nc
    inc idx
    lda idx
    cmp dotn
    bne loop
    rts
}

MACRO CTRIT table
{
    lda table,y
    beq done
    bmi minus
    clc
    lda alo,x
    adc val
    sta alo,x
    bcc done
    inc ahi,x
    bcs done
.minus
    sec
    lda alo,x
    sbc val
    sta alo,x
    bcs done
    dec ahi,x
.done
    inx
}
ENDMACRO

.add_col
{
    lda #0
    sta kbyte
    tax
.loop
    ldy kbyte
    lda (ptr),y
    tay
    CTRIT tr0
    CTRIT tr1
    CTRIT tr2
    CTRIT tr3
    CTRIT tr4
    inc kbyte
    cpx nout
    beq done
    jmp loop
.done
    rts
}

\ ---------------------------------------------------------------- data
.cmd_basic
    EQUS "BASIC", 13

.osw_block
    EQUW linebuf
    EQUB MAXLEN         \ longest line accepted
    EQUB 32
    EQUB 126

.ctx_base
FOR i, 0, K-1
    EQUB i * CTXW
    EQUB i * CTXW + N1
NEXT

.punct
    EQUS ".,?!"

.banner
    EQUB 22, 7
IF BPLUS
    EQUB 134 : EQUS "BeebLM" : EQUB 135 : EQUS "- a neural net for the BBC B+" : EQUB 13
ELSE
    EQUB 134 : EQUS "BeebLM" : EQUB 135 : EQUS "- a neural net in under 32K" : EQUB 13
ENDIF
    EQUB 13
    EQUS "Type to talk. ESCAPE to leave." : EQUB 13
    EQUB 13
    EQUB 0

\ trit lookup: packed byte -> weight for each of its five positions (0, 1 or &FF)
MACRO TRITTAB d
FOR i, 0, 242
    EQUB ((i DIV d) MOD 3) + (((i DIV d) MOD 3) DIV 2) * 253
NEXT
ENDMACRO
.tr0 TRITTAB 1
.tr1 TRITTAB 3
.tr2 TRITTAB 9
.tr3 TRITTAB 27
.tr4 TRITTAB 81

.model_data
.ht0   INCBIN "build/ht0.bin"
.ht1   INCBIN "build/ht1.bin"
.ht2   INCBIN "build/ht2.bin"
.words INCBIN "build/words.bin"
.bqlo  INCBIN "build/bqlo.bin"
.bqhi  INCBIN "build/bqhi.bin"
.bzlo  INCBIN "build/bzlo.bin"
.bzhi  INCBIN "build/bzhi.bin"
.b2lo  INCBIN "build/b2lo.bin"
.b2hi  INCBIN "build/b2hi.bin"
.b3lo  INCBIN "build/b3lo.bin"
.b3hi  INCBIN "build/b3hi.bin"
.wq    INCBIN "build/wq.bin"
.wz    INCBIN "build/wz.bin"
.w1    INCBIN "build/w1.bin"
.w3    INCBIN "build/w3.bin"
IF BPLUS = 0
.w2    INCBIN "build/w2.bin"
ENDIF
.end_of_file

\ ---------------------------------------------------------------- workspace (not saved)
.acc     SKIP H1
.qacc    SKIP H1
.h2      SKIP H2
.qflag   SKIP Q
.alo     SKIP AMAX   \ 16-bit accumulators shared by every layer
.ahi     SKIP AMAX
.clock   SKIP 5
.zv      SKIP HQ
.linebuf SKIP MAXLEN + 2
.nbuf    SKIP MAXLEN + 4
.end_of_ram

PRINT "code", model_data - start, "bytes, model", end_of_file - model_data, "bytes, file ends at", ~end_of_file, ", workspace ends at", ~end_of_ram
IF BPLUS = 0
SAVE "BEEBLM", start, end_of_file, start
ELSE
ASSERT end_of_ram <= &7B00
SAVE "MAIN", start, end_of_file, start

\ ---------------------------------------------------------------- hidden-layer weights, in paged RAM
CLEAR &7C00, &7C00
ORG &8000
GUARD &B000
.w2    INCBIN "build/w2.bin"
.w2_end
SAVE "W2", w2, w2_end, 0, &3000        \ loads at &3000, the loader moves it up

\ ---------------------------------------------------------------- loader
ORG &7B00
.loader
    ldx #LO(cmd_load)
    ldy #HI(cmd_load)
    jsr OSCLI
    lda ROMCPY
    pha
    ora #&80
    sta ROMCPY
    sta ROMSEL
    lda #0
    sta &70
    sta &72
    lda #&30
    sta &71
    lda #&80
    sta &73
    ldx #48             \ 48 pages = 12K
    ldy #0
.loader_copy
    lda (&70),y
    sta (&72),y
    iny
    bne loader_copy
    inc &71
    inc &73
    dex
    bne loader_copy
    pla
    sta ROMCPY
    sta ROMSEL
    ldx #LO(cmd_run)
    ldy #HI(cmd_run)
    jmp OSCLI
.cmd_load
    EQUS "LOAD W2", 13
.cmd_run
    EQUS "RUN MAIN", 13
.loader_end
SAVE "BEEBLM", loader, loader_end, loader
ENDIF
PUTTEXT "boot.txt", "!BOOT", 0
